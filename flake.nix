{
  description = "userspace rcu with an optionally impure clang";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    clang-unwrapped = {
      url = "file://.";
      flake = false;
    };
  };
  outputs = { self, nixpkgs, flake-utils, clang-unwrapped, ...}:
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
      my-wrapped-llvm = pkgs.wrapCCWith rec {
          cc = my-llvm;
          libc = pkgs.stdenv.cc.libc;
          bintools = pkgs.wrapBintoolsWith {
            bintools = pkgs.llvmPackages.bintools-unwrapped;
            libc = pkgs.stdenv.cc.libc;
          };
        };
      mystdenv = (pkgs.overrideCC pkgs.llvmPackages.stdenv my-wrapped-llvm);
      inputs_common = with pkgs; [ autoconf automake libtool ];
    in
    {

      defaultPackage = pkgs.llvmPackages.stdenv.mkDerivation {
        name = "urcu-pure";
        shellHook = ''
          export BUILD_STATIC=1;
        '';
        src = self;
        buildInputs = inputs_common;

        installPhase = ''
          mkdir -p $out/bin
        '';
      };
      impurePackage = mystdenv.mkDerivation {
        name = "urcu-impure";
        shellHook = ''
          export BUILD_STATIC=1;
        '';
        src = self;
        buildInputs = inputs_common ++ (with pkgs; [ gdb ]);

        installPhase = ''
          mkdir -p $out/bin
        '';
      };
    }
  );
}
