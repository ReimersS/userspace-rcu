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
      mkUrcu = { name, stdenv, cflags ? "" }: stdenv.mkDerivation {
        inherit name;
        src = self;
        buildInputs = inputs_common;
        preConfigure = "autoreconf -fiv";
        dontDisableStatic = true;
        configureFlags = [ "--enable-compiler-atomic-builtins" "--disable-shared" ];
        CFLAGS = cflags;
        CXXFLAGS = cflags;
        installPhase = installTests;
      };
    in
    {
      packages.default = mkUrcu { name = "urcu-clang"; stdenv = pkgs.llvmPackages.stdenv; cflags = "-O0"; };

      clangO0Package   = mkUrcu { name = "urcu-clang"; stdenv = pkgs.llvmPackages.stdenv; cflags = "-O0"; };
      clangO3Package   = mkUrcu { name = "urcu-clang"; stdenv = pkgs.llvmPackages.stdenv; cflags = "-O3"; };
      clangirO0Package = mkUrcu { name = "urcu-clangir"; stdenv = orbstdenv; cflags = "-fclangir -O0"; };
      clangirO3Package = mkUrcu { name = "urcu-clangir"; stdenv = orbstdenv; cflags = "-fclangir -O3"; };
      orbO0Package     = mkUrcu { name = "urcu-orb";     stdenv = orbstdenv; cflags = "-fclangir -Xclang -orb -O0"; };
      orbO3Package     = mkUrcu { name = "urcu-orb";     stdenv = orbstdenv; cflags = "-fclangir -Xclang -orb -O3"; };

      devShells.default = pkgs.mkShell {
        packages = [
          #pkgs.jupyter
          (pkgs.python3.withPackages (ps: with ps; [ pandas numpy seaborn matplotlib ]))
        ];
      };

      defaultPackage = self.packages.${system}.default;
      impurePackage = mystdenv.mkDerivation {
        name = "urcu-impure";
        src = self;
        buildInputs = inputs_common ++ (with pkgs; [ gdb ]);
        preConfigure = "autoreconf -fiv";
        dontDisableStatic = true;
        configureFlags = [ "--enable-compiler-atomic-builtins" "--disable-shared" ];
        CFLAGS = "-fclangir -Xclang -orb";
        CXXFLAGS = "-fclangir -Xclang -orb";
        installPhase = installTests;
      };
    }
  );
}
