using ScriptsOfTribute;
using ScriptsOfTribute.AI;
using ScriptsOfTribute.Board;
using ScriptsOfTribute.Serializers;
using System.Diagnostics;
using System.Globalization;
using ScriptsOfTribute.Board.CardAction;
using ScriptsOfTribute.Board.Cards;

namespace Bots;


// Copy of SakkirinaNeural.cs (our current best agent) for cluster data
// generation. One addition only: the exploration-sampling mechanism from
// SakkirinaGen.cs, ported exactly (see ENABLE_EXPLORATION/EXPLORATION_TURNS/
// EXPLORATION_TEMPERATURE and Node.SampleChildByVisits below), so self-play
// between two near-deterministic instances of this search visits more of the
// state space instead of replaying the same line every game. Same search,
// same time budget, same neural evaluator -- do not restructure the search
// here; if it needs fixing, fix it in SakkirinaNeural.cs (or SakkirinaSolo.cs
// for the underlying search) first.
public class SakkirinaGenNeural : AI
{
    // For the first EXPLORATION_TURNS turns, sample the root move
    // proportional to visit count instead of taking BestChild(). After that,
    // behaves exactly like stock SakkirinaNeural.
    const bool ENABLE_EXPLORATION = true;
    const int EXPLORATION_TURNS = 6;

    // Each child's weight is visits^(1/EXPLORATION_TEMPERATURE). At 1.0 this
    // is raw visit-proportional sampling (a no-op vs. the previous
    // behaviour); raising it flattens the distribution toward uniform, for
    // when root visit splits are too concentrated (e.g. [70469, 15, 15]) for
    // proportional sampling to meaningfully diverge from BestChild().
    const double EXPLORATION_TEMPERATURE = 1.0;

    static bool CheckRandomTransition(SeededGameState gameState, SeededGameState newGameState)
    {
        foreach (UniqueCard card0 in newGameState.CurrentPlayer.Hand) {
            bool found = false;
            foreach (UniqueCard card1 in gameState.CurrentPlayer.Hand) {
                if (card0.UniqueId == card1.UniqueId) { found = true; break; }
            }
            if (!found) {
                foreach (UniqueCard card1 in gameState.CurrentPlayer.KnownUpcomingDraws) {
                    if (card0.UniqueId == card1.UniqueId) { found = true; break; }
                }
            }
            if (!found) return true;
        }
        foreach (UniqueCard card0 in newGameState.TavernAvailableCards) {
            bool found = false;
            foreach (UniqueCard card1 in gameState.TavernAvailableCards) {
                if (card0.UniqueId == card1.UniqueId) { found = true; break; }
            }
            if (!found) return true;
        }
        return false;
    }

    class Child
    {
        public Node parent;
        public Move move;
        public double prob;
        public List<Node> nodes;
        public bool stochastic;
        public int selected;

        public double wins;
        public double avgWins;
        public ulong visits;

        public Child(Node parent, Move move, double prob)
        {
            this.wins = 0;
            this.avgWins = 0;
            this.visits = 0;

            this.parent = parent;
            this.move = move;
            this.prob = prob;
            this.selected = 0;
            this.stochastic = false;
            this.nodes = null;
        }

        void AddNode(ulong seed = 0)
        {
            var (newGameState, newPossibleMoves) = (parent.gameState, parent.possibleMoves);
            if (seed == 0) {
                (newGameState, newPossibleMoves) = parent.gameState.ApplyMove(this.move);
            } else {
                (newGameState, newPossibleMoves) = parent.gameState.ApplyMove(this.move, seed);
            }
            bool stochastic = CheckRandomTransition(parent.gameState, newGameState);
            this.nodes.Add(new Node(newGameState, newPossibleMoves));
            if (stochastic) this.stochastic = true; // 全ての遷移先に共通のはず
        }

        public Node SelectChance(SeededRandom rng)
        {
            if (this.nodes is null) {
                this.nodes = new List<Node>();
                this.AddNode(0);
                return this.nodes[0];
            }
            if (!this.stochastic) return this.nodes[0];

            // 不確定の場合遷移先を増やしていく
            if (this.selected == 0) {
                double k = Math.Pow(visits, 0.3);
                if (k >= this.nodes.Count) {
                     // 候補手間で共通のシードを使う
                    if (this.parent.childSeeds is null) this.parent.childSeeds = new List<ulong>();
                    ulong seed;
                    if (this.parent.childSeeds.Count >= this.nodes.Count) {
                        seed = this.parent.childSeeds[this.nodes.Count - 1];
                    } else {
                        seed = (ulong)rng.Next();
                        this.parent.childSeeds.Add(seed);
                    }
                    this.AddNode(seed);
                    this.selected = this.nodes.Count - 1;
                    return this.nodes.Last();
                }
            }
            //foreach (var node in this.nodes) Console.Write("{0} ", node.visits);
            //Console.WriteLine("");

            this.selected -= 1;
            if (this.selected < 0) this.selected = this.nodes.Count - 1;
            return this.nodes[this.selected];
        }

        public void Update(double v)
        {
            this.avgWins = (this.avgWins * this.visits + v) / (this.visits + 1);
            if (this.nodes is null) {
                this.wins = this.avgWins;
            } else {
                double wins = 0;
                ulong visits = 0;
                foreach (var node in this.nodes) {
                    wins += node.wins * node.visits;
                    visits += node.visits;
                }
                this.wins = (wins + this.avgWins * (this.visits + 1 - visits)) / (this.visits + 1);
            }
            this.visits += 1;
        }
    }

    class Node
    {
        public List<Child>? childs;

        public SeededGameState gameState;
        public List<Move>? possibleMoves;
        public bool anyInvalidMoves;
        public List<ulong>? childSeeds;

        public double wins;
        public ulong visits;

        public Node(SeededGameState gameState, List<Move>? possibleMoves)
        {
            this.wins = 0;
            this.visits = 0;

            this.gameState = gameState;
            this.possibleMoves = possibleMoves; // not modified

            this.childs = null;
            this.childSeeds = null;
            if (possibleMoves is not null) { // not turn end
                var moveProbs = new List<(Move move, double prob)>();
                var ruleMove = RootRuleBasedMove(possibleMoves, gameState);
                if (ruleMove is not null) moveProbs.Add((ruleMove, 1.0));
                else if (possibleMoves.Count > 1) {
                    var probs = LogitsToProbs(SimulationPolicy(possibleMoves, gameState));
                    for (int i = 0; i < probs.Count(); i++) moveProbs.Add((possibleMoves[i], probs[i]));
                    moveProbs.OrderBy(x => -x.prob);
                }
                this.childs = moveProbs.ConvertAll<Child>(x => new Child(this, x.Item1, x.Item2));
            }
        }

        public void Update(double v)
        {
            double wins = 0;
            foreach (var child in this.childs) wins = Math.Max(wins, child.wins);
            this.wins = wins;
            this.visits += 1;
        }

        public Child BanditChild(bool root)
        {
            // 探索する手を返す
            double bestScore = -100000;
            int selected = 0;
            int index = 0;
            double logVisits = Math.Log(this.visits + 1);
            double v = (this.wins * this.visits + 0.5) / (this.visits + 1);
            foreach (var child in this.childs) {
                double q = (child.wins * child.visits + v) / (child.visits + 1);
                double p = 0.9 + 0.1 * child.prob;
                double score = q + (root ? 3 : 1) * Math.Sqrt(2 * p * logVisits / (child.visits + 1));
                if (score > bestScore) {
                    bestScore = score;
                    selected = index;
                }
                index++;
            }
            return this.childs[selected];
        }

        public Child BestChild()
        {
            // 最善手を返す
            double bestScore = -100000;
            int selected = 0;
            int index = 0;
            foreach (var child in this.childs) {
                double score = child.wins;
                if (score > bestScore) {
                    bestScore = score;
                    selected = index;
                }
                index++;
            }
            return this.childs[selected];
        }

        // Exploration: sample a root child proportional to
        // visits^(1/EXPLORATION_TEMPERATURE) instead of always taking the
        // argmax. Falls back to BestChild() if no visits have been recorded
        // yet (shouldn't happen given the search always runs at least one
        // iteration, but a zero-visit weighted draw has nothing to sample
        // from).
        //
        // T=1.0 uses the exact same integer arithmetic as the original
        // raw-visits sampling (rather than routing through Math.Pow and a
        // float cumulative threshold) so that it is a bit-for-bit no-op, not
        // just a distributionally-equivalent one -- the same rng.Next() draw
        // must map to the same child it always did, since a same-seed game
        // is used to verify this behaviourally.
        public Child SampleChildByVisits(SeededRandom rng)
        {
            if (EXPLORATION_TEMPERATURE == 1.0) {
                ulong totalVisits = 0;
                foreach (var child in this.childs) totalVisits += child.visits;
                if (totalVisits == 0) return BestChild();

                ulong r = (ulong)rng.Next() % totalVisits;
                ulong cumulative = 0;
                foreach (var child in this.childs) {
                    cumulative += child.visits;
                    if (r < cumulative) return child;
                }
                return this.childs[this.childs.Count - 1];
            }

            var weights = new double[this.childs.Count];
            double totalWeight = 0;
            for (int i = 0; i < this.childs.Count; i++) {
                weights[i] = Math.Pow(this.childs[i].visits, 1.0 / EXPLORATION_TEMPERATURE);
                totalWeight += weights[i];
            }
            if (totalWeight <= 0) return BestChild();

            double r2 = totalWeight * rng.Next() / int.MaxValue;
            double cumWeight = 0;
            for (int i = 0; i < weights.Length; i++) {
                cumWeight += weights[i];
                if (r2 < cumWeight) return this.childs[i];
            }
            return this.childs[this.childs.Count - 1];
        }
    }

    Node? rootNode;
    TimeSpan usedTimeInTurn = TimeSpan.FromSeconds(0);
    TimeSpan TurnTimeout = TimeSpan.FromSeconds(9.8);
    PlayerEnum myPlayerID;
    SeededRandom rng;

    // Counts this bot's own turns within the current game, incremented at
    // each detected turn-start below. Exploration sampling is only active
    // while _turnNumber <= EXPLORATION_TURNS.
    private int _turnNumber = 0;

    // ONNX evaluator + perspective-flip bookkeeping for Evaluate() above,
    // mirroring ISMCTSBot's ValueNetworkEvaluator usage and END_TURN flip
    // tracking.
    private ValueNetworkEvaluator? _evaluator;
    private long _evalFlippedCount = 0;
    private long _evalNotFlippedCount = 0;

    // Throughput telemetry: an unambiguous, directly-counted alternative to
    // inferring iteration counts from the private search tree's visit counts
    // (which broke because ProceedTree re-roots the tree at the end of every
    // Play() call, so a before/after snapshot of rootNode.visits can end up
    // reading two different nodes' counters). _evalNetworkStopwatch times ONLY
    // the EvaluateBoardState call itself, to see what fraction of wall clock
    // the ONNX inference actually accounts for.
    private long _totalEvalCalls = 0;
    private readonly System.Diagnostics.Stopwatch _evalNetworkStopwatch = new System.Diagnostics.Stopwatch();
    private readonly System.Diagnostics.Stopwatch _gameWallClock = new System.Diagnostics.Stopwatch();

    public SakkirinaGenNeural()
    {
        this.rng = new SeededRandom(12345679u);
        this.PrepareForGame();
    }

    void PrepareForGame()
    {
        this.rootNode = null;
        this._turnNumber = 0;
    }

    static double SingleCardValue(Card card)
    {
        var tier = CardTierList.GetCardTier(card.Name);
        return (int)tier * 10;
    }

    static double CardValue(Card card, SerializedPlayer player)
    {
        return SingleCardValue(card); // TODO: 盤面との相関
    }

    static double DeckValue(List<UniqueCard> allCards, bool enemy, double progress)
    {
        double value = 0;
        Dictionary<PatronId, int> potentialComboNumber = new Dictionary<PatronId, int>();
        int writOfCoinCount = 0;
        double cardCountCoef = Math.Sqrt(Math.Max(1.0, allCards.Count / 14.0));

        foreach (var card in allCards) {
            value += SingleCardValue(card) / cardCountCoef;
            if (card.Deck == PatronId.TREASURY) {
                if (card.CommonId == CardId.WRIT_OF_COIN) writOfCoinCount += 1;
            } else {
                if (card.CommonId == CardId.BEWILDERMENT) value -= 30 / cardCountCoef;
                else if (potentialComboNumber.ContainsKey(card.Deck)) potentialComboNumber[card.Deck] += 1;
                else potentialComboNumber[card.Deck] = 1;
            }
        }

        value += (40 - 20 * progress) * writOfCoinCount / cardCountCoef;

        foreach (KeyValuePair<PatronId, int> entry in potentialComboNumber) {
            value += Math.Pow(entry.Value, 1.5); // 最大コンボ数
            value += Math.Pow(entry.Value / (double)allCards.Count, 3) * Math.Min(7, entry.Value) * 20;
        }

        return value;
    }

    static double AgentValue(SerializedAgent agent)
    {
        var tier = CardTierList.GetCardTier(agent.RepresentingCard.Name);
        return 10 * (int)tier + agent.CurrentHp * 3;
    }

    // Original hand-written heuristic, kept so the two evaluators can be A/B'd
    // inside the identical search later. No longer called by Simulate() -- see
    // Evaluate() below, which replaces it as the active evaluator.
    static double EvaluateHeuristic(SeededGameState gameState, PlayerEnum playerID)
    {
        double value = 0;

        // パトロンの選好
        int myPatronFavour = 0;
        int enemyPatronFavour = 0;
        int neutralPatronFavour = 0;
        int myPatronDistance = 0;
        int enemyPatronDistance = 0;
        foreach (var (patron, pId) in gameState.PatronStates.All) {
            if (patron == PatronId.TREASURY) continue;
            if (pId == playerID) {
                myPatronFavour += 1;
                if (patron == PatronId.DUKE_OF_CROWS) value -= 100; // カラスは好意持ちだと使えなくなる
                else if (patron == PatronId.ORGNUM) value += 15;

                if (patron == PatronId.ANSEI) enemyPatronDistance += 1;
                else enemyPatronDistance += 2;
            } else if (pId == PlayerEnum.NO_PLAYER_SELECTED) {
                neutralPatronFavour += 1;
                myPatronDistance += 1;
                enemyPatronDistance += 1;
            } else {
                enemyPatronFavour += 1;
                if (patron == PatronId.DUKE_OF_CROWS) value += 100;
                else if (patron == PatronId.ORGNUM) value -= 15;
                else if (patron == PatronId.ANSEI) value -= 25;

                if (patron == PatronId.ANSEI) myPatronDistance += 1;
                else myPatronDistance += 2;
            }
        }
        if (myPatronFavour >= 4) return 1;

        value += (myPatronFavour - enemyPatronFavour) * 20;
        if (enemyPatronDistance == 1) value -= 3000;
        else if (enemyPatronDistance == 2) value -= 300;
        else if (enemyPatronDistance == 3) value -= 30;
        if (myPatronDistance == 1) value += 50;
        else if (myPatronDistance == 2) value += 5;

        var currentPlayer = playerID == gameState.CurrentPlayer.PlayerID ? gameState.CurrentPlayer : gameState.EnemyPlayer;
        var enemyPlayer = playerID == gameState.CurrentPlayer.PlayerID ? gameState.EnemyPlayer : gameState.CurrentPlayer;

        // プレステージ, パワー
        if (currentPlayer.Prestige >= 80) return 1;
        if (enemyPlayer.Prestige >= 40 && currentPlayer.Prestige < enemyPlayer.Prestige) return 0;

        double progress = Math.Min(Math.Max(currentPlayer.Prestige, enemyPlayer.Prestige), 40) / 40.0;
        double prestigeValue = 10 + progress * 55 + (currentPlayer.Prestige >= 40 ? 10 : 0);

        value -= prestigeValue * 2;
        value += (currentPlayer.Prestige - enemyPlayer.Prestige) * prestigeValue;
        if (enemyPlayer.Prestige == 79) value -= 1000;
        else if (enemyPlayer.Prestige == 78) value -= 300;
        else if (enemyPlayer.Prestige == 77) value -= 100;
        else if (enemyPlayer.Prestige == 76) value -= 30;

        // 場に出ているエージェント
        foreach (SerializedAgent agent in currentPlayer.Agents) {
            value += AgentValue(agent);
        }
        foreach (SerializedAgent agent in enemyPlayer.Agents) {
            value -= AgentValue(agent) * 2 + 40; // 相手が先にエージェントを使える
        }

        // デッキ評価
        List<UniqueCard> allCards = currentPlayer.Hand.Concat(currentPlayer.Played.Concat(currentPlayer.CooldownPile.Concat(currentPlayer.DrawPile))).ToList();
        List<UniqueCard> allCardsEnemy = enemyPlayer.Hand.Concat(enemyPlayer.DrawPile).Concat(enemyPlayer.Played.Concat(enemyPlayer.CooldownPile)).ToList();
        value += DeckValue(allCards, false, progress);
        value -= DeckValue(allCardsEnemy, true, progress);

        // タバーン
        foreach (var card in gameState.TavernAvailableCards) {
            var tier = CardTierList.GetCardTier(card.Name);
            value -= 2 * (int)tier;
        }

        return 1.0 / (1 + Math.Exp(-value / 900.0));
    }

    // Active evaluator: same signature and semantics as EvaluateHeuristic above
    // (a pseudo-win-probability in [0,1] for playerID), backed by the DeepSets
    // value network instead of the hand-written heuristic. The network outputs
    // P(gameState.CurrentPlayer wins), so it must be negated whenever playerID
    // isn't the state's current player -- exactly the same perspective-flip
    // logic as ISMCTSBot's END_TURN handling, tracked the same way so the flip
    // rate can be verified from the log.
    double Evaluate(SeededGameState gameState, PlayerEnum playerID)
    {
        _totalEvalCalls++;
        if (_evaluator == null) return EvaluateHeuristic(gameState, playerID);

        var (nodeFeatures, _, globalFeatures) = FeatureExtractor.ParseState(gameState, false);
        _evalNetworkStopwatch.Start();
        float wp = _evaluator.EvaluateBoardState(nodeFeatures, globalFeatures);
        _evalNetworkStopwatch.Stop();

        bool flipped = playerID != gameState.CurrentPlayer.PlayerID;
        if (flipped) _evalFlippedCount++;
        else _evalNotFlippedCount++;

        return flipped ? (1.0 - wp) : (double)wp;
    }

    // 即出しするカードの集合
    static readonly HashSet<CardId> resourceOnlyCards = new HashSet<CardId> {
        // 資源を得るだけのカード
        // 資源を持っていることがマイナスにならなければ、即出しして良い
        // Treasury
        CardId.WRIT_OF_COIN,
        CardId.GOLD,
        // Hlaalu
        CardId.GOODS_SHIPMENT,
        CardId.LUXURY_EXPORTS,
        // Red Eagle
        CardId.WAR_SONG,
        CardId.MIDNIGHT_RAID,
        // Crow
        CardId.PECK,
        CardId.SCRATCH,
        CardId.MURDER_OF_CROWS,
        // Pellin
        CardId.FORTIFY,
        CardId.THE_PORTCULLIS,
        CardId.REINFORCEMENTS,
        CardId.LEGIONS_ARRIVAL,
        CardId.ARCHERS_VOLLEY,
        CardId.SIEGE_WEAPON_VOLLEY,
        CardId.THE_ARMORY,
        // Rajhin
        CardId.SWIPE,
        CardId.BEWILDERMENT, // no effect
        CardId.POUNCE_AND_PROFIT, // knockout
        CardId.GRAND_LARCENY, // knockout, opp pres -1
        CardId.JARRING_LULLABY, // knockout, opp destroy
        CardId.SHADOWS_SLUMBER, // knockout, opp destroy
        // Orgnum
        CardId.SEA_ELF_RAID,
        CardId.MAORMER_BOARDING_PARTY,
        CardId.SUMMERSET_SACKING,
        CardId.GHOSTSCALE_SEA_SERPENT,
        CardId.SEA_SERPENT_COLOSSUS,
        CardId.SERPENTPROW_SCHOONER,
        CardId.PYANDONEAN_WAR_FLEET,
        // Psijic
        CardId.MAINLAND_INQUIRIES,
        // Ansei
        // Alessia
    };

    static readonly HashSet<CardId> tavernExchangeCards = new HashSet<CardId> {
        // リソースを得て、タバーン交換の能力を持つカード
        // すぐ使って良さそうだけど、買いたいカードを買ってからの方が良いかも
        // Orgnum
        CardId.STORM_SHARK_WAVECALLER,
        CardId.SERPENTGUARD_RIDER,
        // Rajhin
        CardId.SLIGHT_OF_HAND,
        // Psijic
        CardId.PRESCIENCE,
        CardId.PROPHESY,
    };

    static readonly HashSet<CardId> drawingCards = new HashSet<CardId> {
        // 手札から一枚引く効果を持つカード
        // 使えるなら使いたいが、ペリンなど手札の先頭に戻す効果を先に使った方が良い
        // Crow
        CardId.POOL_OF_SHADOW,
        CardId.TOLL_OF_FLESH,
        CardId.TOLL_OF_SILVER,
        CardId.PILFER,
        CardId.PLUNDER,
        CardId.SQUAWKING_ORATORY,
        // Pellin
        CardId.RALLY
    };

    static readonly HashSet<CardId> selectingResourceCards = new HashSet<CardId> {
        // リソースを選択可能なカード
        // 序盤はコイン, 終盤はパワーも重視か 目的の行動に沿うように選ぶ
        // Ansei
        CardId.WAY_OF_THE_SWORD,
        CardId.WARRIOR_WAVE
    };

    static readonly HashSet<CardId> combo3CoinContracts = new HashSet<CardId> {
        // 3コンボの際にコントラクトで絶対に得をするカード
        // Hlaalu
        CardId.KWAMA_EGG_MINE,
        CardId.EBONY_MINE
    };

    static readonly HashSet<CardId> resourceOnlyAgents = new HashSet<CardId> {
        // 資源を得るだけのエージェント
        // 資源を持っていることがマイナスにならなければ、即使用していい
        // Pellin
        CardId.SHIELD_BEARER,
        CardId.BANGKORAI_SENTRIES,
        CardId.KNIGHTS_OF_SAINT_PELIN,
        CardId.BANNERET,
        CardId.KNIGHT_COMMANDER, // heal
        // Crow
        CardId.BLACKFEATHER_KNIGHT,
        // Allesia
        CardId.ALESSIAN_REBEL,
        CardId.MORIHAUS_SACRED_BULL, // knockout, my opes +3
    };

    static readonly HashSet<CardId> drawingAgents = new HashSet<CardId> {
        // 手札から一枚引く効果を持つカード
        // Crow
        CardId.BLACKFEATHER_BRIGAND,
        CardId.BLACKFEATHER_KNAVE
    };

    static readonly HashSet<CardId> zeroCostCards = new HashSet<CardId> {
        // 0コストの初期カード 重要度が低いので捨てることが多い
        CardId.GOLD, // Treasury
        CardId.GOODS_SHIPMENT, // Hlaalu
        CardId.WAR_SONG, // Red Eagle
        CardId.PECK, // Crow
        CardId.ALESSIAN_REBEL, // Allesia
        CardId.FORTIFY, // Pellin
        CardId.SWIPE, // Rahjin
        CardId.MAINLAND_INQUIRIES, // Psijic
        CardId.SEA_ELF_RAID, // Orgnum 1P + 1C
        CardId.BEWILDERMENT // これも入れておく
    };

    // タバーンから買って即使用のコントラクトカード
    //static readonly HashSet<CardId>

    static Move? RootRuleBasedMove(List<Move> moves, SeededGameState gameState, bool root = false)
    {
        // ルートで使用可能なレベルに絶対的なルールベース
        if (moves.Count == 1) return moves[0];

        // 資源を得るだけの行動は即座に行う
        foreach (Move move in moves) {
            if (move.Command == CommandEnum.PLAY_CARD) {
                var card = (move as SimpleCardMove).Card;
                if (resourceOnlyCards.Contains(card.CommonId)) return move;
            } else if (move.Command == CommandEnum.ACTIVATE_AGENT) {
                var card = (move as SimpleCardMove).Card;
                if (resourceOnlyAgents.Contains(card.CommonId)) return move;
            }
        }

        // 自明な選択を行う
        if (gameState.BoardState == BoardState.CHOICE_PENDING ||
            gameState.BoardState == BoardState.PATRON_CHOICE_PENDING) {
            var choiceType = gameState.PendingChoice.ChoiceFollowUp;
            if (choiceType == ChoiceFollowUp.COMPLETE_TREASURY) { // 不要なカードの交換
                foreach (Move move in moves) {
                    var choices = (move as MakeChoiceMove<UniqueCard>).Choices;
                    UniqueCard card = choices[0];
                    if (card.CommonId == CardId.BEWILDERMENT) return move; // 当惑は邪魔
                }
                foreach (Move move in moves) {
                    var choices = (move as MakeChoiceMove<UniqueCard>).Choices;
                    UniqueCard card = choices[0];
                    if (card.CommonId == CardId.GOLD) return move; // 基本的にはGOLD
                }
            } else if (choiceType == ChoiceFollowUp.DESTROY_CARDS) { // 不要なカードの破壊
                foreach (Move move in moves) {
                    var choices = (move as MakeChoiceMove<UniqueCard>).Choices;
                    int bewildermentCount = 0;
                    foreach (UniqueCard card in choices) {
                        if (card.CommonId == CardId.BEWILDERMENT) bewildermentCount++;
                    }
                    if (choices.Count == bewildermentCount) return move; // 当惑は邪魔
                }
            } else if (choiceType == ChoiceFollowUp.DISCARD_CARDS) { // 手札を捨てさせられる
                foreach (Move move in moves) {
                    var choices = (move as MakeChoiceMove<UniqueCard>).Choices;
                    if (choices.Count != 1) continue;
                    UniqueCard card = choices[0];
                    // TODO: そもそも当惑は候補に入らない?
                    if (card.CommonId == CardId.BEWILDERMENT) return move; // 当惑は邪魔
                }
            }
        }

        // コンボが確定して絶対に得をする行動
        if (gameState.BoardState == BoardState.NORMAL) {
            var ComboStates = gameState.ComboStates;
            foreach (KeyValuePair<PatronId, ComboState> combo in ComboStates.All) {
                if (combo.Key == PatronId.HLAALU && combo.Value.CurrentCombo >= 2) {
                    foreach (Move move in moves) {
                        if (move.Command == CommandEnum.BUY_CARD) {
                            var card = (move as SimpleCardMove).Card;
                            if (combo3CoinContracts.Contains(card.CommonId)) return move;
                        }
                    }
                    break;
                }
            }
        }

        if (gameState.BoardState == BoardState.NORMAL) {
            // 勝利確定時に余計な行動をせずに終了
            if (gameState.CurrentPlayer.Prestige >= 80) return Move.EndTurn();

            // パトロン必勝
            PlayerEnum playerID = gameState.CurrentPlayer.PlayerID;
            int myPatronFavour = 0;
            foreach (var (_, pId) in gameState.PatronStates.All) {
                if (pId == playerID) myPatronFavour += 1;
            }
            if (myPatronFavour >= 3) {
                foreach (Move move in moves) {
                    if (move.Command == CommandEnum.CALL_PATRON) {
                        var patronId = (move as SimplePatronMove).PatronId;
                        if (patronId == PatronId.TREASURY) continue;
                        var patronStatus = gameState.PatronStates.GetFor(patronId);
                        if (patronStatus == PlayerEnum.NO_PLAYER_SELECTED ||
                            (patronStatus != playerID && patronId == PatronId.ANSEI)) return move;
                    }
                }
            }
        }

        return null;
    }
    static Move? SimulationRuleBasedMove(List<Move> moves, SeededGameState gameState)
    {
        // シミュレーション中に確率計算を行う前に利用するルールベース
        var move = RootRuleBasedMove(moves, gameState);
        if (move is not null) return move;

        return null;
    }

    static double[] SimulationPolicy(List<Move> moves, SeededGameState gameState)
    {
        var logits = new double[moves.Count];

        // 共通情報
        var currentPlayer = gameState.CurrentPlayer;
        var enemyPlayer = gameState.EnemyPlayer;
        PlayerEnum playerID = currentPlayer.PlayerID;
        int myPatronFavour = 0;
        int enemyPatronFavour = 0;
        int neutralPatronFavour = 0;
        foreach (var (patron, pId) in gameState.PatronStates.All) {
            if (patron == PatronId.TREASURY) continue;
            if (pId == playerID) myPatronFavour += 1;
            else if (pId == PlayerEnum.NO_PLAYER_SELECTED) neutralPatronFavour += 1;
            else enemyPatronFavour += 1;
        }
        List<UniqueCard> myCards = currentPlayer.Hand.Concat(currentPlayer.Played.Concat(currentPlayer.CooldownPile.Concat(currentPlayer.DrawPile))).ToList();
        Dictionary<PatronId, int> myDeckCount = new Dictionary<PatronId, int>();
        foreach (var card in myCards) {
            if (card.Deck == PatronId.TREASURY) continue;
            if (myDeckCount.ContainsKey(card.Deck)) myDeckCount[card.Deck] += 1;
            else myDeckCount[card.Deck] = 1;
        }
        double progress = Math.Min(Math.Max(currentPlayer.Prestige, enemyPlayer.Prestige), 40) / 40.0;
        var currentPlayedSet = new HashSet<CardId>();
        foreach (var card in currentPlayer.Played) currentPlayedSet.Add(card.CommonId);
        bool zeroCostPlayed = false;
        foreach (var cardId in currentPlayedSet) if (zeroCostCards.Contains(cardId)) { zeroCostPlayed = true; break; }

        int index = 0;
        foreach (Move move in moves) {
            double logit = 0;

            if (gameState.BoardState == BoardState.NORMAL) {
                if (move.Command == CommandEnum.END_TURN) {
                    logit -= 10000; // ターンエンドは基本的に最後
                } else if (move.Command == CommandEnum.PLAY_CARD) {
                    var m = move as SimpleCardMove;
                    var card = m.Card;
                    if (resourceOnlyCards.Contains(card.CommonId)) logit += 50;
                    else if (tavernExchangeCards.Contains(card.CommonId)) logit += 40;
                    else if (selectingResourceCards.Contains(card.CommonId)) logit += 30;
                    else if (drawingCards.Contains(card.CommonId)) logit += 20;
                    // コンボ
                    /**/
                } else if (move.Command == CommandEnum.CALL_PATRON) {
                    logit -= 100;
                    var patronId = (move as SimplePatronMove).PatronId;
                    var patronStatus = gameState.PatronStates.GetFor(patronId);
                    // パトロン
                    if (patronId != PatronId.TREASURY) {
                        if (patronStatus != playerID) logit += 0.5; // 好意がないものを優先
                        if (myPatronFavour >= 2 && patronStatus != playerID) logit += 0.5; // パトロン必勝を見据える
                    }
                    // パトロン優先度
                    if (patronId == PatronId.TREASURY) {
                        logit += 5 - progress * 3;
                        if (currentPlayer.Coins == 2) logit += 1;
                        if (!zeroCostPlayed) logit -= 5;
                        else if (currentPlayedSet.Contains(CardId.BEWILDERMENT)) logit += 0.5;
                    } else if (patronId == PatronId.ORGNUM) logit += 3 + (patronStatus == playerID ? 1 : 0); // オルグヌムは好意時に使いたい
                    else if (patronId == PatronId.DUKE_OF_CROWS) logit += (currentPlayer.Coins - 5) + progress * 2.5; // カラスは序盤ダメ
                    else if (patronId == PatronId.ANSEI) logit += 2 - progress;
                    else if (patronId == PatronId.PELIN) logit += 1;
                    else logit += 0.5;

                } else if (move.Command == CommandEnum.BUY_CARD) {
                    logit -= 20;
                    // TODO: まだコインを貯められる場合は保留
                    var card = (move as SimpleCardMove).Card;
                    logit += CardValue(card, currentPlayer) * 0.05; // 強いカード
                    logit += myDeckCount.GetValueOrDefault(card.Deck, 0) * 1; // 同じカードを何枚持っているか
                } else if (move.Command == CommandEnum.ACTIVATE_AGENT) {
                    logit -= 3;
                    var card = (move as SimpleCardMove).Card;
                    if (resourceOnlyAgents.Contains(card.CommonId)) logit += 0.5;
                    else if (drawingAgents.Contains(card.CommonId)) logit += 0.2;
                } else if (move.Command == CommandEnum.ATTACK) {
                    logit -= 2;
                    var card = (move as SimpleCardMove).Card;
                    logit += CardValue(card, enemyPlayer) * 0.2;
                }
            } else if (gameState.BoardState == BoardState.CHOICE_PENDING ||
                       gameState.BoardState == BoardState.PATRON_CHOICE_PENDING) {
                var choiceType = gameState.PendingChoice.ChoiceFollowUp;
                if (choiceType == ChoiceFollowUp.COMPLETE_TREASURY) { // 不要なものを交換
                    var choices = (move as MakeChoiceMove<UniqueCard>).Choices;
                    UniqueCard card = choices is null ? null : choices[0];
                    if (card.CommonId == CardId.BEWILDERMENT) logit += 10000; // 当惑は邪魔
                    else if (card.CommonId == CardId.GOLD) logit += 10;
                    else if (zeroCostCards.Contains(card.CommonId)) logit += 3;
                    logit -= CardValue(card, currentPlayer) * 0.2;
                } else if (choiceType == ChoiceFollowUp.DESTROY_CARDS) {
                    var choices = (move as MakeChoiceMove<UniqueCard>).Choices;
                    if (choices.Count != 0) {
                        foreach (UniqueCard card in choices) {
                            if (card.CommonId == CardId.BEWILDERMENT) logit += 10000; // 当惑は邪魔
                            else if (card.CommonId == CardId.GOLD) logit += 10;
                            else if (zeroCostCards.Contains(card.CommonId)) logit += 3;
                            logit -= CardValue(card, currentPlayer) * 0.2;
                        }
                    }
                } else if (choiceType == ChoiceFollowUp.DISCARD_CARDS) {
                    var choices = (move as MakeChoiceMove<UniqueCard>).Choices;
                    if (choices.Count != 1) logit -= 1000;
                    else {
                        UniqueCard card = choices[0];
                        if (card.CommonId == CardId.BEWILDERMENT) logit += 10000; // 当惑は邪魔
                        else if (card.CommonId == CardId.GOLD) logit += 10;
                        else if (zeroCostCards.Contains(card.CommonId)) logit += 3;
                        logit -= CardValue(card, currentPlayer) * 0.2;
                    }
                } else if (choiceType == ChoiceFollowUp.KNOCKOUT_AGENTS) {
                    var choices = (move as MakeChoiceMove<UniqueCard>).Choices;
                    foreach (UniqueCard card in choices) {
                        SerializedAgent agent = null;
                        foreach (var a in enemyPlayer.Agents) if (a.RepresentingCard.UniqueId == card.UniqueId) { agent = a; break; }
                        if (agent is not null) logit += AgentValue(agent) * 0.2;
                        else {
                            // 自分のエージェント?
                            foreach (var a in currentPlayer.Agents) if (a.RepresentingCard.UniqueId == card.UniqueId) { agent = a; break; }
                            if (agent is not null) logit -= 3; // TODO: これほんとにある?
                        }
                    }
                } else if (choiceType == ChoiceFollowUp.ACQUIRE_CARDS) {
                    var choices = (move as MakeChoiceMove<UniqueCard>).Choices;
                    foreach (var card in choices) logit += CardValue(card, currentPlayer);
                } else if (choiceType == ChoiceFollowUp.REFRESH_CARDS) {
                    var choices = (move as MakeChoiceMove<UniqueCard>).Choices;
                    foreach (var card in choices) logit += CardValue(card, currentPlayer) * 0.3;
                } else if (choiceType == ChoiceFollowUp.COMPLETE_PELLIN) {
                    var choices = (move as MakeChoiceMove<UniqueCard>).Choices;
                    foreach (var card in choices) logit -= CardValue(card, currentPlayer) * 0.3;
                } else if (choiceType == ChoiceFollowUp.COMPLETE_PSIJIC) {
                    var choices = (move as MakeChoiceMove<UniqueCard>).Choices;
                    foreach (var card in choices) logit -= CardValue(card, enemyPlayer) * 0.3;
                } else if (choiceType == ChoiceFollowUp.COMPLETE_HLAALU) {
                    var choices = (move as MakeChoiceMove<UniqueCard>).Choices;
                    foreach (var card in choices) logit += card.Cost * 1.5 - CardValue(card, enemyPlayer) * 0.1;
                }
            }

            logits[index++] = logit;
        }
        return logits;
    }

    static double[] LogitsToProbs(double[] logits)
    {
        double maxValue = -100000;
        for (int i = 0; i < logits.Count(); i++) if (logits[i] > maxValue) maxValue = logits[i];
        double sum = 0;
        for (int i = 0; i < logits.Count(); i++) {
            logits[i] = Math.Exp(logits[i] - maxValue);
            sum += logits[i];
        }
        for (int i = 0; i < logits.Count(); i++) logits[i] /= sum;
        return logits;
    }

    static Move SimulationMove(List<Move> moves, SeededGameState gameState, SeededRandom rng, double temperature = 0)
    {
        Move? tmp = SimulationRuleBasedMove(moves, gameState);
        if (tmp is not null) return tmp;

        var logits = SimulationPolicy(moves, gameState);
        int index = 0;

        double bestScore = -100000;
        for (int i = 0; i < logits.Length; i++) if (logits[i] > bestScore) {
            index = i;
            bestScore = logits[i];
        }

        if (temperature > 0) {
            double sum = 0;
            for (int i = 0; i < logits.Length; i++) {
                logits[i] = Math.Exp((logits[i] - bestScore) / temperature);
                sum += logits[i];
            }
            //for (int i = 0; i < logits.Length; i++) Console.WriteLine("{0} {1}", moves[i], logits[i] / sum);
            double r = sum * rng.Next() / int.MaxValue;
            for (int i = 0; i < logits.Length; i++) {
                sum -= logits[i];
                if (sum <= 0) {
                    index = i;
                    break;
                }
            }
        }
        return moves[index];
    }

    double Simulate(SeededGameState gameState, List<Move> possibleMoves, SeededRandom rng, bool turnEnd = false)
    {
        if (!turnEnd) {
            Move move;
            do {
                move = SimulationMove(possibleMoves, gameState, rng, 1);
                var (newGameState, newPossibleMoves) = gameState.ApplyMove(move);
                gameState = newGameState;
                possibleMoves = newPossibleMoves;
            } while (move.Command != CommandEnum.END_TURN);
        }
        return Evaluate(gameState, myPlayerID);
    }

    double MoveSimulate(SeededGameState gameState, Move move, SeededRandom rng)
    {
        // シミュレーションの最初で乱数を変更?
        var (newGameState, newPossibleMoves) = gameState.ApplyMove(move);//, (ulong)rng.Next());
        return Simulate(newGameState, newPossibleMoves, rng, move.Command == CommandEnum.END_TURN);
    }

    double TreeSearch(Node node, SeededRandom rng, int depth)
    {
        var child = node.BanditChild(depth == 0);

        double value;
        if (child.move.Command == CommandEnum.END_TURN) {
            value = child.wins <= 0 ? MoveSimulate(node.gameState, child.move, rng) : child.wins;
        } else if (child.visits < 1 || depth > 20) {
            value = MoveSimulate(node.gameState, child.move, rng);
        } else {
            var next = child.SelectChance(rng);
            value = TreeSearch(next, rng, depth + 1);
        }

        child.Update(value);
        node.Update(value);
        return value;
    }

    // Defense in depth alongside the HashMove fixes below and the null-move
    // fallback in Play(): root cause of the crash that used to surface here was
    // found -- Play() could pass ProceedTree a null move (see the comment above
    // the fallback in Play() for the full mechanism), which threw inside the
    // inlined AreIsomorphic call with no frame appearing below ProceedTree in a
    // Release stack trace. With that fixed at the source, this catch should
    // never fire again -- kept anyway because tree-reuse detection failing for
    // ANY reason should cost a cache miss, not the game. Falls back to
    // rootNode = null (the method's own existing "no reuse" outcome) on any
    // exception, logging the first occurrence with full details in case
    // something else ever trips it.
    private bool _firstProceedTreeExceptionLogged = false;

    // Counts how often Play()'s null-move fallback below actually fires.
    private long _noIsomorphicMatchCount = 0;
    private bool _firstNoIsomorphicMatchLogged = false;

    void ProceedTree(Move move)
    {
        try
        {
            if (rootNode is null) return;
            if (rootNode.childs is not null &&
                move.Command != CommandEnum.END_TURN) {
                foreach (var child in rootNode.childs) {
                    if (!child.stochastic && child.nodes?.Count == 1) {
                        if (MoveComparer.AreIsomorphic(child.move, move)) {
                            rootNode = child.nodes[0];
                            return;
                        }
                    }
                }
            }
        }
        catch (Exception ex)
        {
            if (!_firstProceedTreeExceptionLogged)
            {
                _firstProceedTreeExceptionLogged = true;
                BotLog.Write($"SakkirinaGenNeural.ProceedTree: caught exception, falling back to no tree reuse: {ex}");
            }
        }
        rootNode = null;
    }

    static void OutputState(GameState gameState)
    {
        var currentPlayer = gameState.CurrentPlayer;
        Console.WriteLine("Prestige {0} (P {1} C {2}) - {3}",
            currentPlayer.Prestige, currentPlayer.Power, currentPlayer.Coins,
            gameState.EnemyPlayer.Prestige);
    }

    // Load the ONNX model for Evaluate() above, mirroring ISMCTSBot.PregamePrepare.
    // Falls back to EvaluateHeuristic (not a silent 0.5) if the model fails to
    // load, since a perfectly good evaluator already exists in this same file.
    //
    // Model path is overridable via the SOT_MODEL_PATH env var (checked
    // first, used verbatim if set -- no existence check before assignment,
    // so a bad override still shows up clearly in the log below and in the
    // load-failure path, rather than silently falling through to the
    // built-in resolution as if nothing were set), so a data-generation run
    // can point at a different model without a rebuild. Falls back to the
    // existing multi-path resolution when unset.
    public override void PregamePrepare()
    {
        _gameWallClock.Restart();

        string modelName = "DeepSetsValueNetwork.onnx";
        string modelPath = modelName;
        string modelSource;

        string? envOverride = Environment.GetEnvironmentVariable("SOT_MODEL_PATH");
        if (!string.IsNullOrEmpty(envOverride))
        {
            modelPath = envOverride;
            modelSource = "SOT_MODEL_PATH env var";
        }
        else
        {
            modelSource = "built-in multi-path resolution";
            string baseDirPath = System.IO.Path.Combine(AppContext.BaseDirectory, modelName);
            if (System.IO.File.Exists(baseDirPath))
                modelPath = baseDirPath;
            else if (System.IO.File.Exists(modelName))
                modelPath = modelName;
            else if (System.IO.File.Exists(System.IO.Path.Combine("..", "Bots", modelName)))
                modelPath = System.IO.Path.Combine("..", "Bots", modelName);
        }

        string resolvedPath = System.IO.Path.GetFullPath(modelPath);
        bool fileExists = System.IO.File.Exists(modelPath);
        BotLog.Write($"SakkirinaGenNeural.PregamePrepare: model path source={modelSource}, " +
                     $"resolved ONNX model path='{resolvedPath}', exists={fileExists}");

        try
        {
            _evaluator = new ValueNetworkEvaluator(modelPath);

            var fileInfo = new System.IO.FileInfo(modelPath);
            string sha256;
            using (var sha = System.Security.Cryptography.SHA256.Create())
            using (var stream = System.IO.File.OpenRead(modelPath))
            {
                sha256 = BitConverter.ToString(sha.ComputeHash(stream)).Replace("-", "").ToLowerInvariant();
            }
            BotLog.Write($"SakkirinaGenNeural.PregamePrepare: ONNX model loaded OK from '{resolvedPath}', " +
                         $"size={fileInfo.Length} bytes, sha256={sha256}");
        }
        catch (Exception e)
        {
            BotLog.Write($"SakkirinaGenNeural.PregamePrepare: FAILED to load ONNX model from '{resolvedPath}' " +
                         $"(exists={fileExists}). Falling back to EvaluateHeuristic. Exception: {e}");
        }
    }

    public override PatronId SelectPatron(List<PatronId> availablePatrons, int round)
    {
        return availablePatrons[rng.Next() % availablePatrons.Count];
    }

    public override Move Play(GameState gameState, List<Move> possibleMoves_, TimeSpan remainingTime)
    {
        myPlayerID = gameState.CurrentPlayer.PlayerID;
        //OutputState(gameState);
        // 合法手のシャッフル
        var possibleMoves = new List<Move>(possibleMoves_);
        for (int i = possibleMoves.Count - 1; i > 0; i--) {
            var j = rng.Next() % (i + 1);
            var tmp = possibleMoves[i];
            possibleMoves[i] = possibleMoves[j];
            possibleMoves[j] = tmp;
        }

        var sgs = gameState.ToSeededGameState((ulong)rng.Next());

        if (gameState.CompletedActions.Count() == 0 ||
            gameState.CompletedActions.Last().Type == CompletedActionType.END_TURN) { // ターン開始時
            //Console.WriteLine("start turn ***");
            usedTimeInTurn = TimeSpan.FromSeconds(0);
            rootNode = null;
            _turnNumber++;
        }

        var move = possibleMoves[0];
        if (possibleMoves.Count == 1) {
            //Console.WriteLine("only move = {0}", move);
            ProceedTree(move);
            return move;
        }

        move = RootRuleBasedMove(possibleMoves, sgs, true);
        if (move is not null) {
            //Console.WriteLine("easy move = {0}", move);
            ProceedTree(move);
            return move;
        }
        if (usedTimeInTurn >= TurnTimeout) {
            move = SimulationMove(possibleMoves, sgs, rng, 0);
            //Console.WriteLine("fast move = {0}", move);
            ProceedTree(move);
            return move;
        }

        // thinking...
        TimeSpan timeForMoveComputation = TimeSpan.FromSeconds(Math.Min(0.65, (TurnTimeout - usedTimeInTurn).TotalSeconds / 4));
        Stopwatch s = new Stopwatch();
        s.Start();
        if (!CheckIfSameGameStateAfterOneMove(rootNode, gameState)) rootNode = null; // check tree reuse
        //else Console.WriteLine("reuse tree {0}", rootNode.visits);
        if (rootNode is null) rootNode = new Node(sgs, possibleMoves);

        while (s.Elapsed < timeForMoveComputation) {
            TreeSearch(rootNode, rng, 0);
        }
        usedTimeInTurn += s.Elapsed;

        // Exploration noise: for the bot's own first EXPLORATION_TURNS turns,
        // sample the root child proportional to visit count instead of
        // always taking the argmax, so self-play between two near-
        // deterministic instances of this search visits more of the state
        // space. After the window, this is exactly BestChild(), i.e. stock
        // behavior.
        var bestChild = (ENABLE_EXPLORATION && _turnNumber <= EXPLORATION_TURNS)
            ? rootNode.SampleChildByVisits(rng)
            : rootNode.BestChild();
        var bestMove = bestChild.move;
        double wp = bestChild.wins;
        //Console.WriteLine("move = {0} wp = {1} in {2} trials", bestMove,
        //    (float)((int)(wp * 10000)) / 100, rootNode.visits);

        // 木の再利用時のために行動はpossibleMovesから選ぶ
        foreach (Move m in possibleMoves) {
            if (MoveComparer.AreIsomorphic(m, bestMove)) { move = m; break; }
        }

        // Root cause of both crash reports: `move` was overwritten to
        // RootRuleBasedMove(...)'s result above and is only reassigned inside
        // the foreach just above -- if bestMove (picked from a tree that may
        // have been built under a different determinization) has no isomorphic
        // match in the CURRENT possibleMoves, move is still null here.
        // ProceedTree(null) then throws inside the inlined AreIsomorphic (no
        // frame appears below ProceedTree in a Release stack trace, which is
        // why this looked like a ProceedTree-local bug), and separately, a null
        // Move returned to the engine makes EndGame's move-history logging
        // throw too. bestMove itself is not a safe substitute -- it may not be
        // legal in the current (re-determinized) state -- so fall back to
        // possibleMoves[0], which always is.
        if (move is null)
        {
            _noIsomorphicMatchCount++;
            if (!_firstNoIsomorphicMatchLogged)
            {
                _firstNoIsomorphicMatchLogged = true;
                BotLog.Write($"SakkirinaGenNeural.Play: bestMove had no isomorphic match in possibleMoves -- " +
                             $"bestMove.Command={bestMove.Command}, bestMove type={bestMove.GetType().FullName}. " +
                             $"Falling back to possibleMoves[0].");
            }
            move = possibleMoves[0];
        }

        ProceedTree(move);
        return move;
    }

    public override void GameEnd(EndGameState state, FullGameState? finalBoardState)
    {
        _gameWallClock.Stop();

        long totalEvals = _evalFlippedCount + _evalNotFlippedCount;
        double flipRate = totalEvals > 0 ? (double)_evalFlippedCount / totalEvals : 0.0;
        BotLog.Write($"SakkirinaGenNeural.GameEnd: evalFlipped={_evalFlippedCount}, evalNotFlipped={_evalNotFlippedCount}, " +
                     $"flipRate={flipRate.ToString("F4", CultureInfo.InvariantCulture)}, reason={state.Reason}, winner={state.Winner}");

        double wallClockSec = _gameWallClock.Elapsed.TotalSeconds;
        double evalsPerSec = wallClockSec > 0 ? _totalEvalCalls / wallClockSec : 0.0;
        double evalNetworkMs = _evalNetworkStopwatch.Elapsed.TotalMilliseconds;
        double fractionInEvalNetwork = wallClockSec > 0 ? (evalNetworkMs / 1000.0) / wallClockSec : 0.0;
        BotLog.Write($"SakkirinaGenNeural.Throughput: totalEvalCalls={_totalEvalCalls}, " +
                     $"wallClockSec={wallClockSec.ToString("F2", CultureInfo.InvariantCulture)}, " +
                     $"evalsPerSec={evalsPerSec.ToString("F1", CultureInfo.InvariantCulture)}, " +
                     $"msInEvaluateBoardState={evalNetworkMs.ToString("F1", CultureInfo.InvariantCulture)}, " +
                     $"fractionOfWallClockInEvaluateBoardState={fractionInEvalNetwork.ToString("F4", CultureInfo.InvariantCulture)}");

        BotLog.Write($"SakkirinaGenNeural.NoIsomorphicMatchFallback: count={_noIsomorphicMatchCount}");

        _evaluator?.Dispose();
        this.PrepareForGame();
    }

    // The following utilities are from BestMCTS3.
    // I would like to express our deepest gratitude to the author.
    class MoveComparer : Comparer<Move>
    {
        // Release inlining collapses these into ProceedTree's frame in a stack
        // trace, making a NullReferenceException here look like it came from
        // there. Each `as` cast below used to be followed by a null-forgiving
        // `!` dereference with no actual check -- if a move's concrete runtime
        // type doesn't match what Command implies, the cast silently yields
        // null and `!` throws, forfeiting the whole game. A hash collision
        // between two unrecognised moves only costs a missed tree-reuse
        // opportunity (harmless), so every site below now falls back to a
        // Command-only hash instead of throwing, logging the first occurrence
        // of each so the actual offending type is known.
        private static bool _firstUnrecognisedPatronMoveLogged = false;
        private static bool _firstUnrecognisedChoiceMoveLogged = false;
        private static bool _firstUnrecognisedCardMoveLogged = false;

        private static ulong CommandOnlyHash(Move x) => 1_000_000_000_000UL * (ulong)x.Command;

        public static ulong HashMove(Move x)
        {
            ulong hash = 0;

            if (x.Command == CommandEnum.CALL_PATRON)
            {
                var mx = x as SimplePatronMove;
                if (mx is not null)
                {
                    hash = (ulong)mx.PatronId;
                }
                else
                {
                    if (!_firstUnrecognisedPatronMoveLogged)
                    {
                        _firstUnrecognisedPatronMoveLogged = true;
                        BotLog.Write($"MoveComparer.HashMove: CALL_PATRON move was not a SimplePatronMove -- " +
                                     $"actual type {x.GetType().FullName}. Falling back to a Command-only hash.");
                    }
                    return CommandOnlyHash(x);
                }
            }
            else if (x.Command == CommandEnum.MAKE_CHOICE)
            {
                var mx = x as MakeChoiceMove<UniqueCard>;
                if (mx is not null)
                {
                    var ids = mx.Choices.Select(card => (ulong)card.CommonId).OrderBy(id => id);
                    foreach (ulong id in ids) hash = hash * 200UL + id;
                }
                else
                {
                    var mxp = x as MakeChoiceMove<UniqueEffect>;
                    if (mxp is not null)
                    {
                        var ids = mxp.Choices.Select(ef => (ulong)ef.Type).OrderBy(type => type);
                        foreach (ulong id in ids) hash = hash * 200UL + id;
                        hash += 1_000_000_000UL;
                    }
                    else
                    {
                        if (!_firstUnrecognisedChoiceMoveLogged)
                        {
                            _firstUnrecognisedChoiceMoveLogged = true;
                            BotLog.Write($"MoveComparer.HashMove: MAKE_CHOICE move was neither " +
                                         $"MakeChoiceMove<UniqueCard> nor MakeChoiceMove<UniqueEffect> -- " +
                                         $"actual type {x.GetType().FullName}. Falling back to a Command-only hash.");
                        }
                        return CommandOnlyHash(x);
                    }
                }
            }
            else if (x.Command != CommandEnum.END_TURN)
            {
                var mx = x as SimpleCardMove;
                if (mx is not null)
                {
                    hash = (ulong)mx.Card.CommonId;
                }
                else
                {
                    if (!_firstUnrecognisedCardMoveLogged)
                    {
                        _firstUnrecognisedCardMoveLogged = true;
                        BotLog.Write($"MoveComparer.HashMove: {x.Command} move was not a SimpleCardMove -- " +
                                     $"actual type {x.GetType().FullName}. Falling back to a Command-only hash.");
                    }
                    return CommandOnlyHash(x);
                }
            }
            return hash + 1_000_000_000_000UL * (ulong)x.Command;
        }

        public override int Compare(Move x, Move y)
        {
            ulong hx = HashMove(x);
            ulong hy = HashMove(y);
            return hx.CompareTo(hy);
        }

        public static bool AreIsomorphic(Move move1, Move move2)
        {
            if (move1.Command != move2.Command) return false; // Speed up
            return HashMove(move1) == HashMove(move2);
        }
    }

    static bool EqualCards(List<UniqueCard> cards0, List<UniqueCard> cards1)
    {
        var diff = new Dictionary<UniqueId, int>();
        foreach (UniqueCard card in cards0) {
            UniqueId uniqueId = card.UniqueId;
            if (diff.ContainsKey(uniqueId)) diff[uniqueId] += 1;
            else diff[uniqueId] = 1;
        }
        foreach (UniqueCard card in cards1) {
            UniqueId uniqueId = card.UniqueId;
            if (diff.ContainsKey(uniqueId)) diff[uniqueId] -= 1;
            else return false;
        }
        return diff.Values.All(n => n == 0);
    }

    static bool CheckIfSameGameStateAfterOneMove(Node node, GameState gameState)
    {
        return node is not null &&
            EqualCards(node.gameState.CurrentPlayer.Hand, gameState.CurrentPlayer.Hand) &&
            EqualCards(node.gameState.TavernAvailableCards, gameState.TavernAvailableCards) &&
            EqualCards(node.gameState.CurrentPlayer.CooldownPile, gameState.CurrentPlayer.CooldownPile) &&
            EqualCards(node.gameState.CurrentPlayer.DrawPile, gameState.CurrentPlayer.DrawPile);
    }
}