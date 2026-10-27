{ strataPkgs }:
{ pkgs, config, lib, ... }:
let
  cfg = config.services.strata;

  serverSettings =
    lib.optionalAttrs (cfg.allowedHosts != [ ]) { allowed_hosts = cfg.allowedHosts; }
    // lib.optionalAttrs (cfg.trustedOrigins != [ ]) { trusted_origins = cfg.trustedOrigins; }
    // lib.optionalAttrs (cfg.corsOrigins != [ ]) { cors_origins = cfg.corsOrigins; }
    // cfg.extraServerConfig;
in
{
  options.services.strata = {
    enable = lib.mkEnableOption "Strata local OpenAI/Anthropic-compatible inference API";

    package = lib.mkPackageOption strataPkgs "strata" { };

    family = lib.mkOption {
      type = lib.types.str;
      default = "swift";
      description = ''
        Model preset family (`swift`, `qwen`, `abliterated`, `coder`, `unsloth`, `gyro`, `qwen36`, `deepseek41`),
        or `"custom"` when using `repo`, `urls`, `ggufDir`, or `modelPaths`.
      '';
    };

    model = lib.mkOption {
      type = lib.types.str;
      default = "IQ2_XS";
      description = ''
        Quantization tag or identifier (e.g. `IQ2_XS`, `Q2_0`, `IQ3_XXS`, `IQ3_S`, `IQ1_M`, `Q4_K_M`).
      '';
    };

    modelName = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      description = "Override the model name reported by `/v1/models`.";
    };

    repo = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      example = "unsloth/Qwen3.6-35B-A3B-GGUF";
      description = "Any Hugging Face repository (`owner/repo`). Overrides the `family` preset repo.";
    };

    revision = lib.mkOption {
      type = lib.types.str;
      default = "main";
      description = "Git branch, tag, or commit hash in the Hugging Face repository.";
    };

    files = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [ ];
      example = [ "Qwen3.6-35B-A3B-Q4_K_M.gguf" ];
      description = "Explicit file path(s) inside `repo` or `ggufDir` (supports single or multi-shard GGUFs).";
    };

    patterns = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [ ];
      example = [ "*IQ2_XS*.gguf" ];
      description = "Glob pattern(s) used to select GGUF files inside `repo` or `ggufDir`.";
    };

    urls = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [ ];
      description = "Direct URL(s) to download GGUF shards from (bypasses Hugging Face API discovery).";
    };

    ggufDir = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      description = "Local directory containing already-downloaded GGUF shard(s).";
    };

    modelPaths = lib.mkOption {
      type = lib.types.listOf lib.types.path;
      default = [ ];
      description = "Direct local file path(s) to existing GGUF shard(s).";
    };

    packDir = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      description = "Direct local path to a pre-built Strata pack directory.";
    };

    mode = lib.mkOption {
      type = lib.types.enum [ "auto" "pack" "gguf" ];
      default = "auto";
      description = ''
        Loading mode: `pack` (builds `iq_pack.py` pack + PLE for Qwen3.8 MoE),
        `gguf` (loads single/sharded GGUF directly), or `auto` (detects from GGUF tensors).
      '';
    };

    mtp = lib.mkOption {
      type = lib.types.enum [ "auto" "on" "off" ];
      default = "auto";
      description = "Enable or disable the MTP speculative decoding draft layer.";
    };

    draftRepo = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      description = "Optional Hugging Face repo for an external speculative decoding draft model.";
    };

    draftFile = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      description = "Filename or glob pattern in `draftRepo`.";
    };

    draftPath = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      description = "Local path to an existing draft GGUF or MTP runtime directory.";
    };

    mmprojRepo = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      description = "Optional Hugging Face repo for a multimodal vision encoder (`mmproj` GGUF).";
    };

    mmprojFile = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      description = "Filename or glob pattern inside `mmprojRepo` (defaults to `*mmproj*.gguf`).";
    };

    mmprojPath = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      description = "Local path to an existing `mmproj` GGUF file.";
    };

    engramRepo = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      description = "Optional Hugging Face repo for DeepSeek V4.1 shared Engram lookup table.";
    };

    engramFile = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      description = "Filename or glob pattern inside `engramRepo`.";
    };

    engramPath = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      description = "Local path to an existing DeepSeek Engram lookup table.";
    };

    context = lib.mkOption {
      type = lib.types.ints.positive;
      default = 32768;
      description = ''
        Context window in tokens (e.g. 8192, 32768, 65536, 131072, 262144, 524288, 1048576).
        From 64K on, KV cache streams from pinned RAM. Above `trainedContext`, YaRN scaling is added automatically.
      '';
    };

    trainedContext = lib.mkOption {
      type = lib.types.ints.positive;
      default = 262144;
      description = "Model's native trained context length threshold for automatic YaRN scaling.";
    };

    host = lib.mkOption {
      type = lib.types.str;
      default = "127.0.0.1";
      description = "Listen address. Use 0.0.0.0 to serve on LAN (requires `apiKeyFile` or `allowedHosts`).";
    };

    port = lib.mkOption {
      type = lib.types.port;
      default = 8080;
    };

    apiKeyFile = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      description = "File holding the API key (loaded via systemd LoadCredential).";
    };

    hfTokenFile = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      description = "File holding a Hugging Face token for gated/private model repositories.";
    };

    allowedHosts = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [ ];
      description = "Allowed HTTP Host headers when running behind a reverse proxy or on LAN.";
    };

    trustedOrigins = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [ ];
      description = "Trusted Origin headers for CSRF protection.";
    };

    corsOrigins = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [ ];
      description = "CORS allowed origins.";
    };

    openFirewall = lib.mkOption {
      type = lib.types.bool;
      default = false;
    };

    gpus = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [ ];
      description = "CUDA device indices (PCI order). Empty picks the GPU with the most VRAM.";
    };

    parallel = lib.mkOption {
      type = lib.types.nullOr lib.types.ints.positive;
      default = null;
      description = "Number of concurrent inference slots (`parallel` in config.json).";
    };

    lazyLoad = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = "Start the HTTP server immediately and defer loading weights into VRAM until the first request.";
    };

    lowRam = lib.mkOption {
      type = lib.types.nullOr lib.types.bool;
      default = null;
      description = "Map experts from disk (`experts.bin`) instead of pinning in RAM. Null auto-detects from system RAM.";
    };

    sampling = lib.mkOption {
      type = lib.types.attrsOf lib.types.anything;
      default = { };
      example = { temperature = 1.0; top_p = 0.95; top_k = 20; };
      description = "Default sampling parameters.";
    };

    mcpServers = lib.mkOption {
      type = lib.types.attrsOf lib.types.anything;
      default = { };
      description = "MCP servers for the web UI chat.";
    };

    extraEngineArgs = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [ ];
      description = "Extra CLI arguments appended to the engine command line (e.g. `[\"--kv-streaming\" \"--resident-experts\"]`).";
    };

    extraServerConfig = lib.mkOption {
      type = lib.types.attrsOf lib.types.anything;
      default = { };
      description = "Additional top-level JSON fields merged into the generated `config.json`.";
    };

    mockEngine = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = "Serve a scripted mock engine instead of the CUDA engine (for VM/smoke tests).";
    };

    prefetchOnStart = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = "Run `strata-prefetch` in `ExecStartPre` to download/prepare weights and emit `config.json`.";
    };
  };

  config = lib.mkIf cfg.enable {
    assertions = [
      {
        assertion = cfg.host == "127.0.0.1" || cfg.host == "localhost" || cfg.apiKeyFile != null || cfg.allowedHosts != [ ];
        message = "services.strata: exposing the API beyond loopback requires apiKeyFile or allowedHosts.";
      }
      {
        assertion = !cfg.mockEngine || !cfg.prefetchOnStart;
        message = "services.strata: mockEngine requires prefetchOnStart = false.";
      }
      {
        assertion = cfg.mockEngine || (cfg.package ? passthru.engine);
        message = "services.strata: package must be the strata-serve bundle (carrying passthru.engine).";
      }
    ];

    systemd.services.strata = {
      description = "Strata local LLM inference API";
      documentation = [ "https://github.com/Niko1221/Strata" ];
      wantedBy = [ "multi-user.target" ];
      wants = [ "network-online.target" ];
      after = [ "network-online.target" ];
      path = [ cfg.package ];

      environment = {
          STRATA_PYTHON = "${cfg.package.passthru.py}/bin/python";
        } // lib.optionalAttrs (cfg.gpus != [ ]) {
          CUDA_DEVICE_ORDER = "PCI_BUS_ID";
          CUDA_VISIBLE_DEVICES = lib.concatStringsSep "," cfg.gpus;
        };

      preStart = lib.mkIf (cfg.prefetchOnStart && !cfg.mockEngine) (toString (
        pkgs.writeShellScript "strata-prestart" (''
          set -euo pipefail
          state="$STATE_DIRECTORY"
          exec ${cfg.package}/bin/strata-prefetch \
            --family ${lib.escapeShellArg cfg.family} \
            --model ${lib.escapeShellArg cfg.model} \
            --mode ${lib.escapeShellArg cfg.mode} \
            --mtp ${lib.escapeShellArg cfg.mtp} \
            --context ${toString cfg.context} \
            --trained-context ${toString cfg.trainedContext} \
            --data-dir "$state" \
            --emit-config "$state/config.json" \
            --engine ${lib.escapeShellArg (toString cfg.package.passthru.engine)}/bin/strata \
            --host ${lib.escapeShellArg cfg.host} \
            --port ${toString cfg.port} \
        '' + lib.optionalString (cfg.modelName != null) ''
            --model-name ${lib.escapeShellArg cfg.modelName} \
        '' + lib.optionalString (cfg.repo != null) ''
            --repo ${lib.escapeShellArg cfg.repo} \
            --revision ${lib.escapeShellArg cfg.revision} \
        '' + lib.concatMapStrings (f: ''--file ${lib.escapeShellArg f} '') cfg.files
           + lib.concatMapStrings (p: ''--pattern ${lib.escapeShellArg p} '') cfg.patterns
           + lib.concatMapStrings (u: ''--url ${lib.escapeShellArg u} '') cfg.urls
           + lib.concatMapStrings (m: ''--model-path ${lib.escapeShellArg (toString m)} '') cfg.modelPaths
           + lib.optionalString (cfg.ggufDir != null) ''
            --gguf-dir ${lib.escapeShellArg (toString cfg.ggufDir)} \
        '' + lib.optionalString (cfg.packDir != null) ''
            --pack-dir ${lib.escapeShellArg (toString cfg.packDir)} \
        '' + lib.optionalString (cfg.draftRepo != null) ''
            --draft-repo ${lib.escapeShellArg cfg.draftRepo} \
        '' + lib.optionalString (cfg.draftFile != null) ''
            --draft-file ${lib.escapeShellArg cfg.draftFile} \
        '' + lib.optionalString (cfg.draftPath != null) ''
            --draft-path ${lib.escapeShellArg (toString cfg.draftPath)} \
        '' + lib.optionalString (cfg.mmprojRepo != null) ''
            --mmproj-repo ${lib.escapeShellArg cfg.mmprojRepo} \
        '' + lib.optionalString (cfg.mmprojFile != null) ''
            --mmproj-file ${lib.escapeShellArg cfg.mmprojFile} \
        '' + lib.optionalString (cfg.mmprojPath != null) ''
            --mmproj-path ${lib.escapeShellArg (toString cfg.mmprojPath)} \
        '' + lib.optionalString (cfg.engramRepo != null) ''
            --engram-repo ${lib.escapeShellArg cfg.engramRepo} \
        '' + lib.optionalString (cfg.engramFile != null) ''
            --engram-file ${lib.escapeShellArg cfg.engramFile} \
        '' + lib.optionalString (cfg.engramPath != null) ''
            --engram-path ${lib.escapeShellArg (toString cfg.engramPath)} \
        '' + lib.optionalString (cfg.lowRam != null) ''
            --low-ram ${if cfg.lowRam then "on" else "off"} \
        '' + lib.optionalString (cfg.hfTokenFile != null) ''
            --hf-token-file "/run/credentials/strata.service/hf-token" \
        '' + lib.optionalString (cfg.apiKeyFile != null) ''
            --api-key-file "/run/credentials/strata.service/api-key" \
        '' + lib.optionalString (cfg.gpus != [ ]) ''
            --gpu ${lib.escapeShellArg (lib.concatStringsSep "," cfg.gpus)} \
        '' + lib.optionalString (cfg.parallel != null) ''
            --parallel ${toString cfg.parallel} \
        '' + lib.optionalString cfg.lazyLoad ''
            --lazy \
        '' + lib.optionalString (cfg.sampling != { }) ''
            --sampling-json ${pkgs.writeText "strata-sampling.json" (builtins.toJSON cfg.sampling)} \
        '' + lib.optionalString (cfg.mcpServers != { }) ''
            --mcp-json ${pkgs.writeText "strata-mcp.json" (builtins.toJSON cfg.mcpServers)} \
        '' + lib.optionalString (serverSettings != { }) ''
            --server-settings-json ${pkgs.writeText "strata-server-settings.json" (builtins.toJSON serverSettings)} \
        '' + lib.concatMapStrings (x: ''--extra-arg ${lib.escapeShellArg x} '') cfg.extraEngineArgs
        )
      ));

      serviceConfig = {
        Type = "simple";
        DynamicUser = true;
        SupplementaryGroups = [ "video" "render" ];
        StateDirectory = "strata";
        StateDirectoryMode = "0750";
        TimeoutStartSec = "0";
        WorkingDirectory = "%S/strata";
        UMask = "0077";
        Restart = "on-failure";
        RestartSec = 10;
        # Required for cudaHostAlloc / mlock expert & KV cache streaming and real-time thread pools
        LimitMEMLOCK = "infinity";
        LimitRTTIME = "infinity";
        LoadCredential =
          (lib.optional (cfg.apiKeyFile != null) "api-key:${cfg.apiKeyFile}")
          ++ (lib.optional (cfg.hfTokenFile != null) "hf-token:${cfg.hfTokenFile}");
        ExecStart =
          if cfg.mockEngine
          then
            "${cfg.package}/bin/strata-serve --engine mock --port ${toString cfg.port} " +
            "--host ${lib.escapeShellArg cfg.host}"
          else
            "${cfg.package}/bin/strata-serve --engine strata --config %S/strata/config.json " +
            "--port ${toString cfg.port}";
        NoNewPrivileges = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        PrivateTmp = true;
        # Allow access to /dev/nvidia* and /dev/dri/*
        PrivateDevices = false;
        DeviceAllow = [
          "/dev/nvidiactl rw"
          "/dev/nvidia-uvm rw"
          "/dev/nvidia-uvm-tools rw"
          "/dev/nvidia-modeset rw"
          "char-nvidia-frontend rw"
          "char-drm rw"
        ];
        ProtectKernelTunables = true;
        ProtectKernelModules = true;
        ProtectControlGroups = true;
        RestrictAddressFamilies = [ "AF_UNIX" "AF_INET" "AF_INET6" ];
        LockPersonality = true;
        MemoryDenyWriteExecute = false;
      };
    };

    networking.firewall.allowedTCPPorts = lib.mkIf cfg.openFirewall [ cfg.port ];
  };
}
