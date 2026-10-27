{
  lib,
  stdenvNoCC,
  makeWrapper,
  python3,
  strataSrc,
  llamaSrc,
  engine,
  version,
  pinChecked ? true,
}:

assert pinChecked;

let
  py = python3.withPackages (ps: with ps; [
    numpy
    jinja2
    regex
    pyyaml
    tqdm
    requests
    pillow
    psutil
  ]);

  share = stdenvNoCC.mkDerivation {
    pname = "strata-share";
    inherit version;
    src = strataSrc;
    phases = [ "installPhase" ];
    installPhase = ''
      mkdir -p $out
      cp -rL ${strataSrc}/. $out/
      chmod -R u+w $out
      rm -rf $out/bench $out/docs $out/ref $out/third_party $out/.gitattributes $out/.gitignore
      mkdir -p $out/third_party/llama.cpp
      cp -rL ${llamaSrc}/gguf-py $out/third_party/llama.cpp/gguf-py
    '';
  };
in
stdenvNoCC.mkDerivation {
  pname = "strata";
  inherit version;
  dontUnpack = true;
  nativeBuildInputs = [ makeWrapper ];

  installPhase = ''
    runHook preInstall
    mkdir -p $out/bin $out/share
    cp -r ${share} $out/share/strata
    chmod -R u+w $out/share/strata
    install -Dm644 ${./prefetch.py} $out/share/strata/prefetch.py

    makeWrapper ${py}/bin/python $out/bin/strata-serve \
      --prefix PATH : "${lib.makeBinPath [ engine ]}" \
      --set STRATA_SHARE "$out/share/strata" \
      --add-flags "$out/share/strata/serve/server.py"

    makeWrapper ${py}/bin/python $out/bin/strata-chat \
      --prefix PATH : "${lib.makeBinPath [ engine ]}" \
      --set STRATA_SHARE "$out/share/strata" \
      --add-flags "$out/share/strata/chat.py"

    makeWrapper ${py}/bin/python $out/bin/strata-prefetch \
      --prefix PATH : "${lib.makeBinPath [ engine ]}" \
      --set STRATA_SHARE "$out/share/strata" \
      --add-flags "$out/share/strata/prefetch.py"

    if [[ -f "$out/share/strata/strata.py" ]]; then
      makeWrapper ${py}/bin/python $out/bin/strata-cli \
        --prefix PATH : "${lib.makeBinPath [ engine ]}" \
        --set STRATA_SHARE "$out/share/strata" \
        --add-flags "$out/share/strata/strata.py"
    fi

    runHook postInstall
  '';

  passthru = {
    inherit engine share py;
    prefetchFlags = { };
  };

  meta = {
    description = "Strata local inference server (OpenAI/Anthropic-compatible API)";
    homepage = "https://github.com/Niko1221/Strata";
    license = lib.licenses.mit;
    platforms = [ "x86_64-linux" ];
    mainProgram = "strata-serve";
  };
}
