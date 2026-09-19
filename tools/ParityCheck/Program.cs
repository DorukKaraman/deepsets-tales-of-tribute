// tools/ParityCheck: a dev/CI-only utility for tools/verify_parity.py.
//
// Reconstructs real ScriptsOfTribute.Serializers.GameState (and, for the
// seeded-check mode, SeededGameState) objects from logged state JSON, then
// runs the REAL Bots.FeatureExtractor.ParseState on them -- not a
// reimplementation of its formulas. This is possible because FeatureExtractor
// only ever reads a narrow, well-defined surface (Deck/Cost/Type/HP/Taunt/
// CommonId on cards; Coins/Power/Prestige/PatronCalls/PlayerID and the card
// piles on players; PatronStates.All), so the reconstructed objects only need
// to be correct on that surface -- not full engine-valid game states.
//
// Three targeted workarounds were needed, all using ONLY public APIs (no
// reflection):
//   - UniqueCard has no simple constructor; built via Card(...).CreateUniqueCopy(),
//     passing an empty effects array (safe: FeatureExtractor calls
//     CardDatabase.GetCardEffects(card.CommonId) for effects, never reads
//     card.Effects itself).
//   - SerializedAgent's only constructor takes a live Agent, which itself only
//     needs a UniqueCard; CurrentHp is matched via Agent.Damage/Heal (both
//     public) rather than set directly.
//   - PatronStates' only constructor takes List<Patron> (live engine objects);
//     its .All dictionary is public and mutable, so it's populated directly
//     from an empty-constructed instance.
//
// Modes:
//   dump <input.json> <output.json>
//     input.json: [{"id": "...", "state": {...raw logged state JSON...}}, ...]
//     output.json: [{"id": "...", "node_matrix": [[99 floats], ...], "global_vector": [19 floats]}, ...]
//
//   seeded-check <input.json>
//     For each input state, derives BOTH a GameState and, from the same
//     underlying board, a SeededGameState (via GameState.ToSeededGameState),
//     runs FeatureExtractor.ParseState on both, and reports whether the
//     ENEMY_UNSEEN node count/CommonId-multiset match. This is a C#-internal
//     check (both overloads of the SAME real method) -- no Python involved --
//     because SeededGameState is exactly what the bot uses inside MCTS, so a
//     mismatch here would never show up in a GameState-only test.
//
//   infer <input.json> <onnx_path> <output.jsonl>
//     For each input state, runs FeatureExtractor.ParseState then the REAL
//     Bots.ValueNetworkEvaluator.EvaluateBoardState against <onnx_path> --
//     the actual C# inference path, not the training-side model. Writes one
//     JSON object per line matching experiments/bots/FeatureDumper.cs's full_*.jsonl
//     schema exactly (game_id/turn/is_terminal/num_nodes/csharp_prob/global/
//     nodes), so experiments/verify_csharp_inference.py reads it completely
//     unmodified -- this replaces that tool's live-game SOT_DUMP_DIR sampler
//     (1-in-50000 evals, no coverage guarantee) with a deterministic run over
//     every sampled state, using the same real code path.

using System.Text.Json;
using System.Text.Json.Serialization;
using Bots;
using ScriptsOfTribute;
using ScriptsOfTribute.Board.Cards;
using ScriptsOfTribute.Serializers;

if (args.Length == 0)
{
    Console.Error.WriteLine("Usage: ParityCheck dump <input.json> <output.json>");
    Console.Error.WriteLine("       ParityCheck seeded-check <input.json>");
    Console.Error.WriteLine("       ParityCheck infer <input.json> <onnx_path> <output.jsonl>");
    return 1;
}

switch (args[0])
{
    case "dump":
        return RunDump(args[1], args[2]);
    case "seeded-check":
        return RunSeededCheck(args[1]);
    case "infer":
        return RunInfer(args[1], args[2], args[3]);
    default:
        Console.Error.WriteLine($"Unknown mode: {args[0]}");
        return 1;
}

int RunDump(string inputPath, string outputPath)
{
    using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(inputPath));
    var results = new List<DumpResult>();

    foreach (JsonElement caseEl in doc.RootElement.EnumerateArray())
    {
        string id = caseEl.GetProperty("id").GetString()!;
        JsonElement stateEl = caseEl.GetProperty("state");

        GameState gs = ReconstructGameState(stateEl);
        var (nodeFeatures, _, globalFeatures) = FeatureExtractor.ParseState(gs);

        int numNodes = nodeFeatures.GetLength(0);
        var nodeMatrix = new float[numNodes][];
        for (int i = 0; i < numNodes; i++)
        {
            nodeMatrix[i] = new float[FeatureExtractor.NODE_DIM];
            for (int j = 0; j < FeatureExtractor.NODE_DIM; j++)
            {
                nodeMatrix[i][j] = nodeFeatures[i, j];
            }
        }

        results.Add(new DumpResult(id, nodeMatrix, globalFeatures));
    }

    var options = new JsonSerializerOptions { WriteIndented = false };
    File.WriteAllText(outputPath, JsonSerializer.Serialize(results, options));
    Console.Error.WriteLine($"Wrote {results.Count} case(s) to {outputPath}");
    return 0;
}

int RunInfer(string inputPath, string onnxPath, string outputPath)
{
    if (!File.Exists(onnxPath))
    {
        Console.Error.WriteLine($"ERROR: onnx model not found at {onnxPath}");
        return 1;
    }

    using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(inputPath));

    ValueNetworkEvaluator evaluator;
    try
    {
        evaluator = new ValueNetworkEvaluator(onnxPath);
    }
    catch (Exception ex)
    {
        Console.Error.WriteLine($"ERROR: could not construct ValueNetworkEvaluator from {onnxPath}: {ex}");
        return 1;
    }

    int count = 0;
    using (evaluator)
    using (var writer = new StreamWriter(outputPath))
    {
        foreach (JsonElement caseEl in doc.RootElement.EnumerateArray())
        {
            JsonElement stateEl = caseEl.GetProperty("state");

            GameState gs = ReconstructGameState(stateEl);
            var (nodeFeatures, _, globalFeatures) = FeatureExtractor.ParseState(gs);
            float csharpProb = evaluator.EvaluateBoardState(nodeFeatures, globalFeatures);

            int numNodes = nodeFeatures.GetLength(0);
            var nodesJagged = new float[numNodes][];
            for (int i = 0; i < numNodes; i++)
            {
                nodesJagged[i] = new float[FeatureExtractor.NODE_DIM];
                for (int j = 0; j < FeatureExtractor.NODE_DIM; j++)
                {
                    nodesJagged[i][j] = nodeFeatures[i, j];
                }
            }

            // Matches experiments/bots/FeatureDumper.cs's full_*.jsonl schema exactly so
            // experiments/verify_csharp_inference.py reads this file unmodified.
            // game_id/turn/is_terminal are placeholders (only used for
            // display in that tool's failure diagnostics, never for the
            // actual comparison) since this harness has no live-game turn
            // context -- count doubles as a stable, distinct game_id.
            var record = new InferRecord(count, 0, false, numNodes, csharpProb, globalFeatures, nodesJagged);
            writer.WriteLine(JsonSerializer.Serialize(record));
            count++;
        }
    }

    Console.Error.WriteLine($"Wrote {count} case(s) to {outputPath}");
    return 0;
}

int RunSeededCheck(string inputPath)
{
    using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(inputPath));
    int checkedCount = 0, mismatchCount = 0;

    foreach (JsonElement caseEl in doc.RootElement.EnumerateArray())
    {
        string id = caseEl.GetProperty("id").GetString()!;
        JsonElement stateEl = caseEl.GetProperty("state");

        GameState gs = ReconstructGameState(stateEl);
        SeededGameState sgs = gs.ToSeededGameState(12345UL);

        // Compare the actual card multisets each overload's ENEMY_UNSEEN loop
        // reads from, not a re-derivation from the output node matrix (the
        // feature vector doesn't carry CommonId directly once effects are
        // looked up, and comparing raw floats would be needlessly fragile).
        List<CardId> gsUnseen = gs.EnemyPlayer.HandAndDraw.Select(c => c.CommonId).OrderBy(x => x).ToList();
        List<CardId> sgsUnseen = sgs.EnemyPlayer.Hand.Concat(sgs.EnemyPlayer.DrawPile)
            .Select(c => c.CommonId).OrderBy(x => x).ToList();

        // Also confirm FeatureExtractor's own node COUNT at LOC_ENEMY_UNSEEN
        // agrees with that multiset size for both overloads -- this is what
        // would actually catch a "loop over DrawPile but forgot Hand" bug in
        // FeatureExtractor itself (Bots/src/DeepSetsCore.cs), as opposed to a
        // bug in this harness's reconstruction.
        var (gsNodes, _, _) = FeatureExtractor.ParseState(gs);
        var (sgsNodes, _, _) = FeatureExtractor.ParseState(sgs);
        int gsUnseenNodeCount = CountAtLocation(gsNodes, locSlot: 8);
        int sgsUnseenNodeCount = CountAtLocation(sgsNodes, locSlot: 8);

        checkedCount++;
        bool multisetMatch = gsUnseen.SequenceEqual(sgsUnseen);
        bool nodeCountMatch = gsUnseenNodeCount == sgsUnseenNodeCount &&
                               gsUnseenNodeCount == gsUnseen.Count && sgsUnseenNodeCount == sgsUnseen.Count;

        if (!multisetMatch || !nodeCountMatch)
        {
            mismatchCount++;
            Console.WriteLine($"MISMATCH id={id}:");
            Console.WriteLine($"  GameState:       {gsUnseen.Count} cards in EnemyPlayer.HandAndDraw, {gsUnseenNodeCount} ENEMY_UNSEEN nodes -- [{string.Join(",", gsUnseen)}]");
            Console.WriteLine($"  SeededGameState: {sgsUnseen.Count} cards in Hand+DrawPile, {sgsUnseenNodeCount} ENEMY_UNSEEN nodes -- [{string.Join(",", sgsUnseen)}]");
        }
        else
        {
            Console.WriteLine($"OK id={id}: {gsUnseen.Count} ENEMY_UNSEEN cards, identical CommonId multiset and node count in both overloads " +
                               $"(SeededGameState split: {sgs.EnemyPlayer.Hand.Count} in Hand, {sgs.EnemyPlayer.DrawPile.Count} in DrawPile -- both loops genuinely exercised)");
        }
    }

    Console.WriteLine();
    Console.WriteLine($"seeded-check: {checkedCount} state(s) checked, {mismatchCount} mismatch(es)");
    return mismatchCount == 0 ? 0 : 1;
}

int CountAtLocation(float[,] nodeFeatures, int locSlot)
{
    int numNodes = nodeFeatures.GetLength(0);
    int count = 0;
    for (int i = 0; i < numNodes; i++)
    {
        if (nodeFeatures[i, 90 + locSlot] == 1.0f) count++;
    }
    return count;
}

GameState ReconstructGameState(JsonElement stateEl)
{
    JsonElement cpEl = stateEl.GetProperty("CurrentPlayer");
    JsonElement epEl = stateEl.GetProperty("EnemyPlayer");

    SerializedPlayer currentPlayer = ParsePlayerFull(cpEl, ParseCardList(cpEl, "Hand"), ParseCardList(cpEl, "DrawPile"));

    // EnemyPlayer in the logged JSON is a FairSerializedEnemyPlayer, which
    // merges Hand+DrawPile into "HandAndDraw". Reconstructing SerializedPlayer
    // needs separate Hand/DrawPile lists, but the split point is unrecoverable
    // (that's exactly what FairSerializedEnemyPlayer's serialization threw
    // away). This doesn't affect the "dump" mode's correctness --
    // FairSerializedEnemyPlayer.HandAndDraw recomputes as
    // DrawPile.Concat(Hand).OrderBy(CommonId) regardless of how the two are
    // split. But an arbitrary non-degenerate split (alternating cards between
    // the two piles, rather than dumping everything into one) matters for
    // "seeded-check": if everything landed in DrawPile, Hand would stay
    // empty and a FeatureExtractor bug that forgot to iterate one of the
    // two piles in the SeededGameState overload would go undetected.
    List<UniqueCard> enemyHandAndDraw = ParseCardList(epEl, "HandAndDraw");
    var enemyHand = new List<UniqueCard>();
    var enemyDraw = new List<UniqueCard>();
    for (int i = 0; i < enemyHandAndDraw.Count; i++)
    {
        (i % 2 == 0 ? enemyHand : enemyDraw).Add(enemyHandAndDraw[i]);
    }
    SerializedPlayer enemyPlayer = ParsePlayerFull(epEl, hand: enemyHand, drawPile: enemyDraw);

    var patronStates = new PatronStates(new List<Patron>());
    foreach (JsonProperty prop in stateEl.GetProperty("PatronStates").GetProperty("All").EnumerateObject())
    {
        PatronId patronId = Enum.Parse<PatronId>(prop.Name);
        PlayerEnum favor = (PlayerEnum)prop.Value.GetInt32();
        patronStates.All[patronId] = favor;
    }

    List<UniqueCard> tavernAvailable = ParseCardList(stateEl, "TavernAvailableCards");

    var fullGameState = new FullGameState(
        currentPlayer, enemyPlayer, patronStates,
        tavernAvailableCards: tavernAvailable,
        tavernCards: new List<UniqueCard>(), // never read by FeatureExtractor
        currentSeed: 0UL);

    return new GameState(fullGameState);
}

SerializedPlayer ParsePlayerFull(JsonElement playerEl, List<UniqueCard> hand, List<UniqueCard> drawPile)
{
    PlayerEnum playerId = (PlayerEnum)playerEl.GetProperty("PlayerID").GetInt32();
    List<UniqueCard> played = ParseCardList(playerEl, "Played");
    List<UniqueCard> cooldown = ParseCardList(playerEl, "CooldownPile");
    List<SerializedAgent> agents = ParseAgentList(playerEl);
    int power = playerEl.GetProperty("Power").GetInt32();
    uint patronCalls = playerEl.TryGetProperty("PatronCalls", out var pc) ? pc.GetUInt32() : 0u;
    int coins = playerEl.GetProperty("Coins").GetInt32();
    int prestige = playerEl.GetProperty("Prestige").GetInt32();

    return new SerializedPlayer(playerId, hand, drawPile, cooldown, played, agents, power, patronCalls, coins, prestige);
}

List<UniqueCard> ParseCardList(JsonElement parent, string propertyName)
{
    var result = new List<UniqueCard>();
    if (!parent.TryGetProperty(propertyName, out JsonElement arr)) return result;
    foreach (JsonElement cardEl in arr.EnumerateArray())
    {
        result.Add(ParseCard(cardEl));
    }
    return result;
}

UniqueCard ParseCard(JsonElement cardEl)
{
    string name = cardEl.GetProperty("Name").GetString() ?? "";
    PatronId deck = (PatronId)cardEl.GetProperty("Deck").GetInt32();
    CardId commonId = (CardId)cardEl.GetProperty("CommonId").GetInt32();
    int cost = cardEl.GetProperty("Cost").GetInt32();
    CardType type = (CardType)cardEl.GetProperty("Type").GetInt32();
    int hp = cardEl.GetProperty("HP").GetInt32();
    bool taunt = cardEl.GetProperty("Taunt").GetBoolean();

    var card = new Card(name, deck, commonId, cost, type, hp, Array.Empty<ComplexEffect?>(), hash: -1, family: null, taunt, copies: 1);
    return card.CreateUniqueCopy();
}

List<SerializedAgent> ParseAgentList(JsonElement parent)
{
    var result = new List<SerializedAgent>();
    if (!parent.TryGetProperty("Agents", out JsonElement arr)) return result;
    foreach (JsonElement agentEl in arr.EnumerateArray())
    {
        UniqueCard card = ParseCard(agentEl.GetProperty("RepresentingCard"));
        int currentHp = agentEl.GetProperty("CurrentHp").GetInt32();
        bool activated = agentEl.GetProperty("Activated").GetBoolean();

        var agent = new Agent(card); // CurrentHp starts at card.HP
        int damage = card.HP - currentHp;
        if (damage > 0) agent.Damage(damage);
        else if (damage < 0) agent.Heal(-damage);
        if (activated) agent.MarkActivated();

        result.Add(new SerializedAgent(agent));
    }
    return result;
}

record DumpResult(
    [property: JsonPropertyName("id")] string Id,
    [property: JsonPropertyName("node_matrix")] float[][] NodeMatrix,
    [property: JsonPropertyName("global_vector")] float[] GlobalVector);

// Field names/order match experiments/bots/FeatureDumper.cs's full_*.jsonl schema exactly.
record InferRecord(
    [property: JsonPropertyName("game_id")] int GameId,
    [property: JsonPropertyName("turn")] int Turn,
    [property: JsonPropertyName("is_terminal")] bool IsTerminal,
    [property: JsonPropertyName("num_nodes")] int NumNodes,
    [property: JsonPropertyName("csharp_prob")] float CsharpProb,
    [property: JsonPropertyName("global")] float[] Global,
    [property: JsonPropertyName("nodes")] float[][] Nodes);
