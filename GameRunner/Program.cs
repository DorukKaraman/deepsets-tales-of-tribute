using System.CommandLine;
using System.Diagnostics;
using GameRunner;
using Bots;
using ScriptsOfTribute;
using ScriptsOfTribute.AI;
using ScriptsOfTribute.Board;
using ScriptsOfTribute.Serializers;
using ScriptsOfTributeGRPC;
using System.CommandLine.Parsing;
using System.CommandLine.Invocation;
using System.Runtime.Loader;
using System.IO.Compression;
using System.Text.Json;
using System.Text.Json.Serialization;

var currentDirectory = new DirectoryInfo(AppContext.BaseDirectory);
var botsDirectory = Path.Combine(currentDirectory.FullName, "Bots");

var aiType = typeof(AI);
var externalBotType = typeof(ExternalAIAdapter);
var botDlls = Directory.Exists(botsDirectory)
    ? Directory.GetFiles(botsDirectory, "*.dll")
    : Array.Empty<string>();

List<Type> allBots = botDlls
    .Select(f => AssemblyLoadContext.Default.LoadFromAssemblyPath(f))
    .SelectMany(a => a.GetTypes())
    .Where(t => aiType.IsAssignableFrom(t) && !t.IsInterface && !t.IsAbstract)
    .ToList();

BotInfo? cachedBot = null;
var returnValue = 0;
#region Options and arguments

var noOfRunsOption = CreateOption<int>("--runs", "Number of games to run.", 1, "-n");
var threadsOption = CreateOption<int>("--threads", "Number of CPU threads to use.", 1, "-t");
var logsOption = CreateOption<LogsEnabled>("--enable-logs", "Enable logging.", LogsEnabled.NONE, "-l");
var seedOption = CreateOption<ulong?>("--seed", "Specify RNG seed.", null, "-s");
var logFileDestination = CreateLogFileOption("--log-destination", "Directory for log files.", "-d");
var timeoutOption = CreateOption<int>("--timeout", "Game timeout in seconds.", 30, "-to");
var clientPortOption = CreateOption<int>("--client-port", "Base client port for gRPC bots.", 50000, "-cp");
var serverPortOption = CreateOption<int>("--server-port", "Base server port for gRPC bots.", 49000, "-sp");
var patronsOption = CreateOption<string[]>("--patrons", "Allowed patrons for this run. Use 'default' (alias: 'all') for standard set (all selectable, no TREASURY).", Array.Empty<string>(), "-p");
var logTrainingDataOption = CreateOption<bool>("--log-training-data", "Log per-turn (state, action) pairs from BOTH bots to sharded gzip JSONL files for value-network training.", false, "-ltd");
var dataDirOption = CreateOption<string>("--data-dir", "Output directory for training-data shards. Used with --log-training-data.", "training_data", "-dd");

var bot1NameArgument = CreateBotArgument("bot1", "Name of the first bot or command.");
var bot2NameArgument = CreateBotArgument("bot2", "Name of the second bot or command.");

var mainCommand = new RootCommand("A game runner for bots.")
{
    noOfRunsOption,
    threadsOption,
    logsOption,
    logFileDestination,
    seedOption,
    timeoutOption,
    clientPortOption,
    serverPortOption,
    patronsOption,
    logTrainingDataOption,
    dataDirOption,
    bot1NameArgument,
    bot2NameArgument,
};

#endregion

#region Bot logic
BotInfo? FindBot(string name, out string? errorMessage)
{
    errorMessage = null;

    if (name.StartsWith("cmd:"))
    {
        var parts = name[4..].Split(' ', 2);
        return new ExternalBotInfo
        {
            BotType = typeof(ExternalAIAdapter),
            ProgramName = parts[0],
            FileName = parts.Length > 1 ? parts[1] : null
        };
    }
    if (name.StartsWith("grpc:"))
    {
        return new gRPCBotInfo
        {
            BotName = name[5..],
            BotType = typeof(gRPCBot),
            HostName = "localhost",
            ClientPort = 50000,
            ServerPort = 49000
        };
    }

    return FindInternalBot(name, out errorMessage) ?? throw new Exception(errorMessage);
}

BotInfo? FindInternalBot(string name, out string? errorMessage)
{
    errorMessage = null;
    var botInfo = new LocalBotInfo()
    {
        BotName = name,
    };
    bool findByFullName = name.Contains('.');
    if (cachedBot is not null && (findByFullName ? cachedBot.BotFullName : cachedBot.BotName) == name)
    {
        return cachedBot;
    }

    var botCount = allBots.Count(t => (findByFullName ? t.FullName : t.Name) == name);

    if (botCount == 0)
    {
        errorMessage = $"Bot {name} not found in any DLLs. List of bots found:\n";
        errorMessage += string.Join('\n', allBots.Select(b => b.FullName));
        return null;
    }

    if (botCount > 1 && !findByFullName)
    {
        errorMessage = "More than one bots with the same name found. Please, specify full name of the target bot: <namespace>.Name. Bots found:\n";
        errorMessage += string.Join('\n', allBots.Select(b => b.FullName));
        return null;
    }

    else if (botCount > 1 && findByFullName)
    {
        errorMessage = "More than one bots with the same full name found. This means you have different DLLs with the same namespaces and bot names.\n" +
                       "This use case is not yet supported. List of all found bots:\n";
        errorMessage += string.Join('\n', allBots.Select(b => b.FullName));
        return null;
    }

    botInfo.BotType = allBots.First(t => (findByFullName ? t.FullName : t.Name) == name);
    cachedBot = botInfo;

    if (cachedBot.BotType.GetConstructor(Type.EmptyTypes) is null)
    {
        errorMessage = $"Bot {name} bot can't be instantiated as it doesn't provide a parameterless constructor.";
    }

    return cachedBot;
}

BotInfo? ParseBotArg(ArgumentResult arg)
{
    if (arg.Tokens.Count != 1)
    {
        arg.ErrorMessage = "Bot name must be a single token.";
        return null;
    }

    var bot = FindBot(arg.Tokens[0].Value, out var errorMessage);
    if (errorMessage is not null)
    {
        arg.ErrorMessage = errorMessage;
        return null;
    }

    return bot!;
}
#endregion

#region Prepare game

ScriptsOfTribute.AI.ScriptsOfTribute PrepareGame(
    AI bot1,
    AI bot2,
    LogsEnabled enableLogs, ulong seed, LogFileNameProvider? logProvider, int timeout, IEnumerable<string>? patronTokens
)
{
    var game = new ScriptsOfTribute.AI.ScriptsOfTribute(bot1, bot2, TimeSpan.FromSeconds(timeout), patronTokens)
    {
        Seed = seed,
    };

    switch (enableLogs)
    {
        case LogsEnabled.P1:
            game.P1LoggerEnabled = true;
            break;
        case LogsEnabled.P2:
            game.P2LoggerEnabled = true;
            break;
        case LogsEnabled.NONE:
            break;
        case LogsEnabled.BOTH:
            game.P1LoggerEnabled = true;
            game.P2LoggerEnabled = true;
            break;
        default:
            throw new ArgumentOutOfRangeException(nameof(enableLogs), enableLogs, null);
    }

    if (logProvider is not null)
    {
        var (p1LogDest, p2LogDest) = logProvider.GetForPlayers(seed, game.P1LoggerEnabled, game.P2LoggerEnabled);
        game.P1LogTarget = p1LogDest;
        game.P2LogTarget = p2LogDest;
    }

    return game;
}

#endregion

#region Main command handler

mainCommand.SetHandler((InvocationContext context) =>
{
    int runs = context.ParseResult.GetValueForOption(noOfRunsOption);
    int threads = context.ParseResult.GetValueForOption(threadsOption);
    LogsEnabled logs = context.ParseResult.GetValueForOption(logsOption);
    LogFileNameProvider? logProvider = context.ParseResult.GetValueForOption(logFileDestination);
    ulong? seed = context.ParseResult.GetValueForOption(seedOption);
    int timeout = context.ParseResult.GetValueForOption(timeoutOption);
    int baseClientPort = context.ParseResult.GetValueForOption(clientPortOption);
    int baseServerPort = context.ParseResult.GetValueForOption(serverPortOption);
    IEnumerable<string>? patrons = context.ParseResult.GetValueForOption(patronsOption);
    bool logTrainingData = context.ParseResult.GetValueForOption(logTrainingDataOption);
    string dataDir = context.ParseResult.GetValueForOption(dataDirOption) ?? "training_data";
    BotInfo? bot1Info = context.ParseResult.GetValueForArgument(bot1NameArgument);
    BotInfo? bot2Info = context.ParseResult.GetValueForArgument(bot2NameArgument);

    if (bot1Info is null || bot2Info is null)
    {
        Console.Error.WriteLine("ERROR: Bots were not parsed correctly.");
        returnValue = -1;
        return;
    }

    string baseHost = "localhost";

    if (!ValidateInputs(threads, timeout)) return;
    ulong actualSeed = seed ?? (ulong)new Random().NextInt64();

    if (threads == 1)
        RunSingleThreaded(runs, bot1Info, bot2Info, logs, logProvider, actualSeed, timeout, baseClientPort, baseServerPort, patrons, logTrainingData, dataDir, baseHost);
    else
        RunMultiThreaded(runs, threads, bot1Info, bot2Info, logs, logProvider, actualSeed, timeout, baseClientPort, baseServerPort, patrons, logTrainingData, dataDir, baseHost);
});

void RunSingleThreaded(
    int runs,
    BotInfo bot1Info,
    BotInfo bot2Info,
    LogsEnabled enableLogs,
    LogFileNameProvider? logFileNameProvider,
    ulong actualSeed,
    int timeout,
    int baseClientPort,
    int baseServerPort,
    IEnumerable<string>? patrons,
    bool logTrainingData,
    string dataDir,
    string baseHost = "localhost"
)
{
    Console.WriteLine($"Running {runs} games - {bot1Info.BotName} vs {bot2Info.BotName}");
    var counter = new GameEndStatsCounter();
    var timeMeasurements = new long[runs];
    var granularWatch = new Stopwatch();
    var currentSeed = actualSeed;

    if (bot1Info is gRPCBotInfo grpcBotInfo1)
    {
        grpcBotInfo1.HostName = baseHost;
        grpcBotInfo1.ClientPort = baseClientPort;
        grpcBotInfo1.ServerPort = baseServerPort;
    }

    if (bot2Info is gRPCBotInfo grpcBotInfo2)
    {
        grpcBotInfo2.HostName = baseHost;
        grpcBotInfo2.ClientPort = baseClientPort+1;
        grpcBotInfo2.ServerPort = baseServerPort+1;
    }

    var innerBot1 = bot1Info.CreateBotInstance();
    var innerBot2 = bot2Info.CreateBotInstance();
    AI bot1 = innerBot1;
    AI bot2 = innerBot2;

    DataLoggingWrapper? bot1Wrapper = null;
    DataLoggingWrapper? bot2Wrapper = null;
    int processId = Process.GetCurrentProcess().Id;
    int gameCounter = 0;

    if (logTrainingData)
    {
        Directory.CreateDirectory(dataDir);
        string matchupSlug = SanitizeForFileName($"{bot1Info.BotName}_vs_{bot2Info.BotName}");
        string shard1Path = Path.Combine(dataDir, $"{matchupSlug}_pid{processId}_w0_bot1.jsonl.gz");
        string shard2Path = Path.Combine(dataDir, $"{matchupSlug}_pid{processId}_w0_bot2.jsonl.gz");
        bot1Wrapper = new DataLoggingWrapper(innerBot1, shard1Path, PlayerEnum.PLAYER1);
        bot2Wrapper = new DataLoggingWrapper(innerBot2, shard2Path, PlayerEnum.PLAYER2);
        bot1 = bot1Wrapper;
        bot2 = bot2Wrapper;
        Console.WriteLine($"Training-data logging enabled. Shards:\n  {shard1Path}\n  {shard2Path}");
    }

    for (var i = 0; i < runs; i++)
    {
        var game = PrepareGame(bot1, bot2, enableLogs, currentSeed, logFileNameProvider, timeout, patrons);
        currentSeed += 1;

        if (logTrainingData)
        {
            string gameId = $"{processId}_0_{gameCounter++}";
            bot1Wrapper!.StartNewGame(gameId);
            bot2Wrapper!.StartNewGame(gameId);
        }

        granularWatch.Reset();
        granularWatch.Start();
        var (endReason, _) = game.Play();
        granularWatch.Stop();

        if (logTrainingData)
        {
            bot1Wrapper!.FinishGame(endReason);
            bot2Wrapper!.FinishGame(endReason);
        }

        // One machine-readable line per game, with the EXACT GameEndReason.
        // GameEndStatsCounter's aggregate (below) buckets TURN_TIMEOUT,
        // INCORRECT_MOVE, BOT_EXCEPTION, INTERNAL_ERROR and both
        // PATRON_SELECTION_* reasons together as "other factors", which is
        // fine for a self-play data run but not for a benchmark: at a 2s
        // per-turn budget a game lost to a timeout is not a game lost to
        // play, and the two have to be reported separately. Parsed by
        // tools/benchmark_cluster.py (GAME_END_REASON_PATTERN).
        Console.WriteLine($"GAME_END_REASON: {endReason.Reason} WINNER: {endReason.Winner}");

        if (endReason.Reason == ScriptsOfTribute.Board.GameEndReason.BOT_EXCEPTION)
            Console.WriteLine(endReason);
        timeMeasurements[i] = granularWatch.ElapsedMilliseconds;
        counter.Add(endReason);
    }

    if (innerBot1 is gRPCBot grpcBot1)
    {
        grpcBot1.CloseConnection();
    }
    if (innerBot2 is gRPCBot grpcBot2)
    {
        grpcBot2.CloseConnection();
    }
    Console.WriteLine($"\nInitial seed used: {actualSeed}");
    Console.WriteLine($"Total time taken: {timeMeasurements.Sum()}ms");
    Console.WriteLine($"Average time per game: {timeMeasurements.Average()}ms");
    Console.WriteLine("\nStats from the games played:");
    Console.WriteLine(counter.ToString());
}

void RunMultiThreaded(
    int runs,
    int noOfThreads,
    BotInfo bot1Info,
    BotInfo bot2Info,
    LogsEnabled enableLogs,
    LogFileNameProvider? logFileNameProvider,
    ulong actualSeed,
    int timeout,
    int baseClientPort,
    int baseServerPort,
    IEnumerable<string>? patrons,
    bool logTrainingData,
    string dataDir,
    string baseHost = "localhost"
)
{
    Console.WriteLine($"Running {runs} games with {noOfThreads} threads.");

    if (logTrainingData) Directory.CreateDirectory(dataDir);

    var gamesPerThread = runs / noOfThreads;
    var gamesPerThreadRemainder = runs % noOfThreads;
    var threads = new Task<List<ScriptsOfTribute.Board.EndGameState>>[noOfThreads];

    List<ScriptsOfTribute.Board.EndGameState> PlayGames(
        int amount,
        BotInfo bot1Info,
        BotInfo bot2Info,
        int threadNo,
        ulong seed,
        IEnumerable<string>? patronsLocal)
    {
        var results = new ScriptsOfTribute.Board.EndGameState[amount];
        var timeMeasurements = new long[amount];
        var watch = new Stopwatch();

        var innerBot1 = bot1Info.CreateBotInstance();
        var innerBot2 = bot2Info.CreateBotInstance();
        AI bot1 = innerBot1;
        AI bot2 = innerBot2;

        DataLoggingWrapper? bot1Wrapper = null;
        DataLoggingWrapper? bot2Wrapper = null;
        int processId = Process.GetCurrentProcess().Id;
        int gameCounter = 0;

        if (logTrainingData)
        {
            string matchupSlug = SanitizeForFileName($"{bot1Info.BotName}_vs_{bot2Info.BotName}");
            string shard1Path = Path.Combine(dataDir, $"{matchupSlug}_pid{processId}_w{threadNo}_bot1.jsonl.gz");
            string shard2Path = Path.Combine(dataDir, $"{matchupSlug}_pid{processId}_w{threadNo}_bot2.jsonl.gz");
            bot1Wrapper = new DataLoggingWrapper(innerBot1, shard1Path, PlayerEnum.PLAYER1);
            bot2Wrapper = new DataLoggingWrapper(innerBot2, shard2Path, PlayerEnum.PLAYER2);
            bot1 = bot1Wrapper;
            bot2 = bot2Wrapper;
            Console.WriteLine($"Thread #{threadNo} training-data shards:\n  {shard1Path}\n  {shard2Path}");
        }

        for (var i = 0; i < amount; i++)
        {
            var game = PrepareGame(bot1, bot2, enableLogs, seed, logFileNameProvider, timeout, patronsLocal);
            seed += 1;

            if (logTrainingData)
            {
                string gameId = $"{processId}_{threadNo}_{gameCounter++}";
                bot1Wrapper!.StartNewGame(gameId);
                bot2Wrapper!.StartNewGame(gameId);
            }

            watch.Reset();
            watch.Start();
            var (endReason, _) = game.Play();
            watch.Stop();

            if (logTrainingData)
            {
                bot1Wrapper!.FinishGame(endReason);
                bot2Wrapper!.FinishGame(endReason);
            }

            results[i] = endReason;
            timeMeasurements[i] = watch.ElapsedMilliseconds;
        }
        if (innerBot1 is gRPCBot grpcBot1)
        {
            grpcBot1.CloseConnection();
        }
        if (innerBot2 is gRPCBot grpcBot2)
        {
            grpcBot2.CloseConnection();
        }
        Console.WriteLine($"Thread #{threadNo} finished. Total: {timeMeasurements.Sum()}ms, average: {timeMeasurements.Average()}ms.");
        return results.ToList();
    }

    var watch = Stopwatch.StartNew();
    var currentSeed = actualSeed;

    for (var i = 0; i < noOfThreads; i++)
    {
        var additionalGames = gamesPerThreadRemainder-- > 0 ? 1 : 0;
        var gamesToPlay = gamesPerThread + additionalGames;
        var threadNo = i;
        var currentSeedCopy = currentSeed;
        var bot1ThreadInfo = bot1Info is gRPCBotInfo grpcBot1
            ? new gRPCBotInfo
            {
                BotName = grpcBot1.BotName,
                BotType = grpcBot1.BotType,
                HostName = grpcBot1.HostName,
                ClientPort = baseClientPort + i,
                ServerPort = baseServerPort + i,
            }
            : bot1Info;

        var bot2ThreadInfo = bot2Info is gRPCBotInfo grpcBot2
            ? new gRPCBotInfo
            {
                BotName = grpcBot2.BotName,
                BotType = grpcBot2.BotType,
                HostName = grpcBot2.HostName,
                ClientPort = baseClientPort + noOfThreads + i,
                ServerPort = baseServerPort + noOfThreads + i,
            }
            : bot2Info;
        threads[i] = Task.Factory.StartNew(() => PlayGames(gamesToPlay, bot1ThreadInfo, bot2ThreadInfo, threadNo, currentSeedCopy, patrons));
        currentSeed += (ulong)gamesToPlay;
    }

    Task.WaitAll(threads);

    var timeTaken = watch.ElapsedMilliseconds;

    var counter = new GameEndStatsCounter();
    threads.SelectMany(t => t.Result).ToList().ForEach(counter.Add);

    Console.WriteLine($"\nInitial seed used: {actualSeed}");
    Console.WriteLine($"Total time taken: {timeTaken}ms");
    Console.WriteLine("\nStats from the games played:");
    Console.WriteLine(counter.ToString());
}

#endregion

#region Helpers

Option<T> CreateOption<T>(string name, string description, T defaultValue, string alias = "")
{
    var option = new Option<T>(name, () =>  defaultValue, description);
    option.AddAlias(alias);
    return option;
}
    

Option<LogFileNameProvider?> CreateLogFileOption(string name, string description, string alias)
{
    var option = new Option<LogFileNameProvider?>(
         name: "--log-destination",
        description: "Log to files with names 'directory/<seed_bot.log>' instead of standard output. Specify the directory here.",
        isDefault: true,
        parseArgument: result =>
        {
            if (result.Tokens.Count == 0)
            {
                return null;
            }

            var dirName = result.Tokens.Single().Value;
            var dir = Directory.CreateDirectory(dirName);
            return new LogFileNameProvider(dir);
        }
    )
    {
        IsRequired = false,
    };
    option.AddAlias("-d");
    return option;
}

Argument<BotInfo?> CreateBotArgument(string name, string description) =>
    new(name, description: description, parse: ParseBotArg);

string SanitizeForFileName(string s)
{
    var invalid = Path.GetInvalidFileNameChars();
    var chars = s.Select(c => invalid.Contains(c) || c == ' ' ? '_' : c).ToArray();
    return new string(chars);
}

bool ValidateInputs(int threads, int timeout)
{
    if (threads < 1)
    {
        Console.Error.WriteLine("ERROR: Can't use less than 1 thread.");
        returnValue = -1;
        return false;
    }

    if (timeout < 0)
    {
        Console.Error.WriteLine("ERROR: Time limit can't be negative.");
        returnValue = -1;
        return false;
    }
    return true;
}

#endregion

mainCommand.Invoke(args);
return returnValue;

#region Data Logging

// Prevents the JSON serializer from stepping on the engine's broken
// PossibleCards/PossibleEffects getters on SerializedChoice.
public class SafeChoiceConverter : JsonConverter<SerializedChoice>
{
    public override SerializedChoice Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
        => throw new NotImplementedException();

    public override void Write(Utf8JsonWriter writer, SerializedChoice value, JsonSerializerOptions options)
    {
        writer.WriteStartObject();

        object? cards = null;
        object? effects = null;

        try { cards = value.PossibleCards; } catch { }
        try { effects = value.PossibleEffects; } catch { }

        if (cards != null)
        {
            writer.WritePropertyName("PossibleCards");
            JsonSerializer.Serialize(writer, cards, options);
        }
        if (effects != null)
        {
            writer.WritePropertyName("PossibleEffects");
            JsonSerializer.Serialize(writer, effects, options);
        }

        writer.WriteEndObject();
    }
}

// Wraps an inner bot to intercept every Play() call and buffer the
// pre-move GameState + chosen move, then dump the buffer to a
// process/worker/matchup-sharded gzip JSONL file at game end -- one JSON
// object per line, tagged with a unique game_id and this wrapper's fixed
// player identity so downstream training can pool records from both
// bots (and both PlayerID perspectives) in a matchup.
public class DataLoggingWrapper : AI
{
    private readonly AI _innerBot;
    private readonly string _logFilePath;
    private readonly PlayerEnum _player;
    private readonly List<(string stateJson, string moveStr)> _turnLogs = new();
    private readonly JsonSerializerOptions _jsonOptions;
    private string _currentGameId = "";

    // Clean, well-defined win/loss outcomes only. Everything else
    // (INCORRECT_MOVE, TURN_TIMEOUT, PATRON_SELECTION_TIMEOUT,
    // PATRON_SELECTION_FAILURE, TURN_LIMIT_EXCEEDED, INTERNAL_ERROR,
    // BOT_EXCEPTION, PREPARE_TIME_EXCEEDED) either has no genuine winner or
    // reflects a malfunction rather than a strategic outcome, so it must not
    // be used as a training label. An allowlist (not a denylist of error
    // reasons) means a future GameEndReason value is excluded by default
    // instead of silently slipping through as "clean".
    private static readonly HashSet<GameEndReason> CleanEndReasons = new()
    {
        GameEndReason.PRESTIGE_OVER_40_NOT_MATCHED,
        GameEndReason.PRESTIGE_OVER_80,
        GameEndReason.PATRON_FAVOR,
    };

    public DataLoggingWrapper(AI innerBot, string logFilePath, PlayerEnum player)
    {
        _innerBot = innerBot;
        _logFilePath = logFilePath;
        _player = player;

        _jsonOptions = new JsonSerializerOptions {
            IncludeFields = true // Player hands and other hidden state live in fields, not properties.
        };
        _jsonOptions.Converters.Add(new SafeChoiceConverter());
    }

    public void StartNewGame(string gameId) => _currentGameId = gameId;

    public override ScriptsOfTribute.Move Play(GameState gameState, List<ScriptsOfTribute.Move> possibleMoves, TimeSpan remainingTime)
    {
        var chosenMove = _innerBot.Play(gameState, possibleMoves, remainingTime);

        string stateJson = JsonSerializer.Serialize(gameState, _jsonOptions);
        _turnLogs.Add((stateJson, chosenMove.ToString() ?? ""));

        return chosenMove;
    }

    // Writes the buffered turns with this wrapper's own perspective's
    // outcome if the game ended cleanly; otherwise discards them. Either way
    // the buffer is empty afterward, ready for the next game.
    public void FinishGame(ScriptsOfTribute.Board.EndGameState endState)
    {
        if (CleanEndReasons.Contains(endState.Reason))
            FlushToFile(endState.Winner == _player);
        else
            DiscardBuffer();
    }

    public void DiscardBuffer() => _turnLogs.Clear();

    // Opens the file in Append mode and wraps a fresh GZipStream per call,
    // producing concatenated gzip members rather than one continuous
    // stream -- Python's gzip module reads these transparently as one
    // logical stream (verified empirically, not just assumed).
    public void FlushToFile(bool didWin)
    {
        int outcome = didWin ? 1 : 0;
        string gameIdJson = JsonSerializer.Serialize(_currentGameId);
        int playerValue = (int)_player;

        using (var fileStream = new FileStream(_logFilePath, FileMode.Append))
        using (var gzipStream = new GZipStream(fileStream, CompressionMode.Compress))
        using (var sw = new StreamWriter(gzipStream))
        {
            foreach (var (stateJson, moveStr) in _turnLogs)
            {
                string moveJson = JsonSerializer.Serialize(moveStr);
                sw.WriteLine($"{{\"outcome\": {outcome}, \"game_id\": {gameIdJson}, \"player\": {playerValue}, " +
                             $"\"data\": {{\"state\": {stateJson}, \"action\": {moveJson}}}}}");
            }
        }
        _turnLogs.Clear();
    }

    public override void GameEnd(ScriptsOfTribute.Board.EndGameState state, FullGameState? finalBoardState)
    {
        _innerBot.GameEnd(state, finalBoardState);
    }

    public override PatronId SelectPatron(List<PatronId> availablePatrons, int round)
    {
        return _innerBot.SelectPatron(availablePatrons, round);
    }

    public override void PregamePrepare()
    {
        _innerBot.PregamePrepare();
    }
}

#endregion
