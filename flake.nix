{
  description = "userspace rcu with an optionally impure clang";

  nixConfig.extra-sandbox-paths = [ "/scratch" ];

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    clang-unwrapped = {
      url = "file://.";
      flake = false;
    };
    clang-orb = {
      url = "git+file:///scratch/sebastian/llvm-orb";
      flake = true;
    };
  };
  outputs = { self, nixpkgs, flake-utils, clang-unwrapped, clang-orb, ...}:
  flake-utils.lib.eachDefaultSystem (system:
    let
      pkgs = import nixpkgs { inherit system; };

      my-llvm = pkgs.stdenv.mkDerivation {
        name = "impure-clang";
        src = null;
        dontUnpack = true;
        installPhase = ''
          mkdir -p $out/bin
          for bin in ${toString (builtins.attrNames (builtins.readDir (clang-unwrapped.outPath+"/build/bin/")))}; do
          cat > $out/bin/$bin <<EOF
          #!${pkgs.runtimeShell}
          exec "$(cat ${clang-unwrapped.outPath+"/.pwd"})/build/bin/$bin" "\$@"
          EOF
          chmod +x $out/bin/$bin
          done
        '';
        passthru.isClang = true;
      };
      my-wrapped-llvm = pkgs.llvmPackages.clang.override { cc = my-llvm; };
      mystdenv = pkgs.overrideCC pkgs.llvmPackages.stdenv my-wrapped-llvm;

      filteredSrc = builtins.path {
        path = self;
        name = "urcu-src";
        filter = path: _type:
          builtins.match ".*\\.(py|json)$" path == null;
      };

      orb-cc = clang-orb.packages.${system}.default;
      orb-wrapped = pkgs.llvmPackages.clang.override { cc = orb-cc; };
      orbstdenv = pkgs.overrideCC pkgs.llvmPackages.stdenv orb-wrapped;

      inputs_common = with pkgs; [ autoconf automake libtool ];
      installTests = ''
        make install
        cp -r tests $out/
        {
          echo "export URCU_TESTS_SRCDIR=$out/tests"
          echo "export URCU_TESTS_BUILDDIR=$out/tests"
          echo "export URCU_TESTS_TIME_BIN=time"
        } > $out/tests/utils/env.sh
      '';
      mkUrcu = { name, stdenv, cflags ? "", synthesis ? false }: stdenv.mkDerivation {
        inherit name;
        src = filteredSrc;
        buildInputs = inputs_common;
        preConfigure = ''
          autoreconf -fiv

          # CC timing wrapper: logs elapsed time per compilation to .cc-times/
          mkdir -p .cc-times
          real_cc="$CC"
          timesdir="$(pwd)/.cc-times"
          wrapper="$(pwd)/.cc-wrapper"
          cat > "$wrapper" <<WRAPPER
#!${pkgs.runtimeShell}
src=""
ofile=""
for arg; do
  case "\$arg" in
    *.c|*.cc|*.cpp) src="\$arg";;
    *.o) ofile="\$arg";;
  esac
done
# Per-target synth log directory: automake names objects
# "target-source.o" when custom CFLAGS exist, else "source.o".
if [ -n "\$ofile" ] && [ -n "\$ORB_SYNTH_LOG_BASE" ]; then
  obase=\$(basename "\$ofile" .o)
  # "test_urcu_mb-test_urcu" -> target="test_urcu_mb"
  # "test_urcu" -> target="test_urcu"
  case "\$obase" in
    *-*) target="''${obase%%-*}";;
    *)   target="\$obase";;
  esac
  export ORB_SYNTH_LOG="\$ORB_SYNTH_LOG_BASE/\$target"
  mkdir -p "\$ORB_SYNTH_LOG"
fi
t0=\$(date +%s%N)
$real_cc "\$@"
rc=\$?
t1=\$(date +%s%N)
if [ -n "\$src" ]; then
  elapsed_ms=\$(( (t1 - t0) / 1000000 ))
  base=\$(basename "\$src")
  obase=\$(basename "\$ofile" .o 2>/dev/null)
  echo "\$base \$obase \$elapsed_ms" >> $timesdir/times.log
fi
exit \$rc
WRAPPER
          chmod +x "$wrapper"
          export CC="$wrapper"
        '' + (if synthesis then ''
          export ORB_SYNTH_LOG_BASE="$(pwd)/.synth-logs"
          mkdir -p "$ORB_SYNTH_LOG_BASE"
        '' else "");
        dontDisableStatic = true;
        enableParallelBuilding = true;
        configureFlags = [ "--enable-compiler-atomic-builtins" "--disable-shared" ];
        CFLAGS = "-march=armv8.1-a+rcpc ${cflags}";
        CXXFLAGS = "-march=armv8.1-a+rcpc ${cflags}";
        installPhase = installTests + ''
          if [ -f .cc-times/times.log ]; then
            mkdir -p $out/cc-times
            cp .cc-times/times.log $out/cc-times/
          fi
        '' + (if synthesis then ''
          if [ -d .synth-logs ]; then
            cp -r .synth-logs $out/synth
          fi
        '' else "");
      };
      fCosts = [1 333 666 999];

    in
    {
      packages.default = mkUrcu { name = "urcu-clang0"; stdenv = orbstdenv; cflags = "-O0"; };

      clangO0Package   = mkUrcu { name = "urcu-clang0"; stdenv = orbstdenv; cflags = "-O0"; };
      clangO3Package   = mkUrcu { name = "urcu-clang3"; stdenv = orbstdenv; cflags = "-O3"; };
      clangirO0Package = mkUrcu { name = "urcu-clangir0"; stdenv = orbstdenv; cflags = "-fclangir -O0"; };
      clangirO3Package = mkUrcu { name = "urcu-clangir3"; stdenv = orbstdenv; cflags = "-fclangir -O3"; };
    } // (builtins.listToAttrs (builtins.concatMap (fc: [
      { name = "orb-O0-fc${toString fc}";
        value = mkUrcu { name = "urcu-orb-O0-fc${toString fc}"; stdenv = orbstdenv; synthesis = true;
                         cflags = "-fclangir -Xclang -orb -Xclang -orb-fence-cost-base=${toString fc} -O0"; }; }
      { name = "orb-O3-fc${toString fc}";
        value = mkUrcu { name = "urcu-orb-O3-fc${toString fc}"; stdenv = orbstdenv; synthesis = true;
                         cflags = "-fclangir -Xclang -orb -Xclang -orb-fence-cost-base=${toString fc} -O3"; }; }
    ]) fCosts)) // {

      "naive-O0" = mkUrcu { name = "urcu-naive-O0"; stdenv = orbstdenv; synthesis = true;
                             cflags = "-fclangir -Xclang -naive-orb -O0"; };
      "naive-O3" = mkUrcu { name = "urcu-naive-O3"; stdenv = orbstdenv; synthesis = true;
                             cflags = "-fclangir -Xclang -naive-orb -O3"; };
    } // {

      devShells.default = pkgs.mkShell {
        packages = [
          #pkgs.jupyter
          (pkgs.python3.withPackages (ps: with ps; [ pandas numpy seaborn matplotlib jupyter ]))
        ];
      };

      packages.orb-cc = orb-wrapped;

      defaultPackage = self.packages.${system}.default;
      impurePackage = mystdenv.mkDerivation {
        name = "urcu-impure";
        src = filteredSrc;
        buildInputs = inputs_common ++ (with pkgs; [ gdb ]);
        preConfigure = "autoreconf -fiv";
        dontDisableStatic = true;
        configureFlags = [ "--enable-compiler-atomic-builtins" "--disable-shared" ];
        CFLAGS = "-march=armv8.1-a+rcpc -fclangir -Xclang -orb";
        CXXFLAGS = "-march=armv8.1-a+rcpc -fclangir -Xclang -orb";
        installPhase = installTests;
      };
      impureNaivePackage = mystdenv.mkDerivation {
        name = "urcu-impure-naive";
        src = filteredSrc;
        buildInputs = inputs_common ++ (with pkgs; [ gdb ]);
        preConfigure = ''
          autoreconf -fiv
          export ORB_SYNTH_LOG="$(pwd)/.synth-logs"
          mkdir -p "$ORB_SYNTH_LOG"
        '';
        dontDisableStatic = true;
        configureFlags = [ "--enable-compiler-atomic-builtins" "--disable-shared" ];
        CFLAGS = "-march=armv8.1-a+rcpc -fclangir -Xclang -naive-orb";
        CXXFLAGS = "-march=armv8.1-a+rcpc -fclangir -Xclang -naive-orb";
        installPhase = installTests + ''
          if [ -d .synth-logs ]; then
            mkdir -p $out/synth
            cp .synth-logs/* $out/synth/ 2>/dev/null || true
          fi
        '';
      };
    }
  );
}
