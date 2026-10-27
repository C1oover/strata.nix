{
  lib,
  stdenv,
  cmake,
  ninja,
  autoAddDriverRunpath,
  cudaPackages,
  src,
  llamaSrc,
  version,
  ggmlPin,
  pinChecked ? true,
  # Support Ampere (86) and Ada (89) by default, or override with e.g. "86"
  cudaArch ? "86",
  enableAvx512 ? false,
}:

assert pinChecked;

let
  archList = lib.splitString ";" cudaArch;
  archJson = builtins.toJSON (map lib.toInt archList);
in
cudaPackages.backendStdenv.mkDerivation {
  pname = "strata-engine";
  inherit version src;
  strictDeps = true;

  nativeBuildInputs = [
    cmake
    ninja
    cudaPackages.cuda_nvcc
    autoAddDriverRunpath
  ];

  buildInputs = [
    cudaPackages.cudatoolkit
    cudaPackages.cuda_cudart
    cudaPackages.libcublas
  ];

  cmakeFlags = [
    "-DCMAKE_GENERATOR=Ninja"
    "-DCMAKE_BUILD_TYPE=Release"
    "-DSTRATA_ENABLE_CUDA=ON"
    "-DSTRATA_BUILD_TESTS=OFF"
    "-DCMAKE_CUDA_ARCHITECTURES=${cudaArch}"
    "-DCMAKE_CUDA_COMPILER=${cudaPackages.cuda_nvcc}/bin/nvcc"
    "-DSTRATA_GGML_DIR=${llamaSrc}"
    "-DGGML_NATIVE=OFF"
    "-DGGML_AVX=ON"
    "-DGGML_AVX2=ON"
    "-DGGML_FMA=ON"
    "-DGGML_F16C=ON"
    "-DGGML_BMI2=ON"
    "-DGGML_AVX512=${if enableAvx512 then "ON" else "OFF"}"
  ];

  env = {
    PIN_ASSERT = toString pinChecked;
    GGML_PIN_ASSERT = ggmlPin;
    CUDAToolkit_ROOT = "${cudaPackages.cudatoolkit}";
    CUDAToolkit_INCLUDE_ROOT = "${cudaPackages.cudatoolkit}";
    CUDAToolkit_LIBRARY_ROOT = "${cudaPackages.cudatoolkit}";
  };

  buildPhase = ''
    runHook preBuild
    ninja -C . -j$NIX_BUILD_CORES strata strata-device
    # Build optional targets (e.g. mtmd / vision / deepseek binaries) if present in upstream build.ninja
    for target in strata-deepseek strata-mtmd mtmd; do
      if ninja -C . -t targets all 2>/dev/null | grep -q "^''${target}:"; then
        ninja -C . -j$NIX_BUILD_CORES "$target"
      fi
    done
    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall
    mkdir -p $out/bin $out/lib
    bin=$(find "$NIX_BUILD_TOP" -maxdepth 4 -type f -executable -name strata | head -1)
    if [[ -z "$bin" ]]; then
      echo "strata binary not found under $NIX_BUILD_TOP" >&2
      exit 1
    fi
    install -Dm755 "$bin" $out/bin/strata

    for optBin in strata-device strata-deepseek strata-mtmd; do
      found=$(find "$NIX_BUILD_TOP" -maxdepth 4 -type f -executable -name "$optBin" | head -1)
      [[ -n "$found" ]] && install -Dm755 "$found" "$out/bin/$optBin"
    done

    find "$NIX_BUILD_TOP" -maxdepth 4 -type f -name "lib*mtmd*.so*" -exec cp -P {} $out/lib/ \;

    visionMode="none"
    if [[ -f "$out/bin/strata-mtmd" ]] || compgen -G "$out/lib/lib*mtmd*.so*" > /dev/null; then
      visionMode="mtmd"
    fi

    cat > $out/bin/BUILD.json <<EOF
{
  "version": "${version}",
  "archs": ${archJson},
  "ptx": false,
  "source": "nix",
  "vision": "$visionMode"
}
EOF
    runHook postInstall
  '';

  passthru = {
    inherit cudaArch enableAvx512;
  };

  meta = {
    description = "Strata CUDA inference engine (sm_${lib.replaceStrings [";"] ["_"] cudaArch})";
    homepage = "https://github.com/Niko1221/Strata";
    license = lib.licenses.mit;
    platforms = [ "x86_64-linux" ];
    mainProgram = "strata";
  };
}
