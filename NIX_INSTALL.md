
add glib and libGL to nix config

and all these:

      cudaPackages.cuda_cccl
      cudaPackages.cuda_cudart
      cudaPackages.cuda_cupti
      cudaPackages.cuda_nvcc
      cudaPackages.cuda_nvml_dev
      cudaPackages.cuda_nvrtc
      cudaPackages.cuda_nvtx
      cudaPackages.cusparselt
      cudaPackages.libcublas
      cudaPackages.libcufft
      cudaPackages.libcufile
      cudaPackages.libcurand
      cudaPackages.libcusolver
      cudaPackages.libcusparse

fix cv2
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(nix eval --raw --expr 'let pkgs = import <nixpkgs> {}; in "${pkgs.libGL}/lib"' --impure --extra-experimental-features "nix-command flakes")":"$(nix eval --raw --expr 'let pkgs = import <nixpkgs> {}; in "${pkgs.glib.out}/lib"' --impure --extra-experimental-features "nix-command flakes")

alternatively

programs.nix-ld = {
   enable = true;
   libraries = options.programs.nix-ld.libraries.default ++ (
     with pkgs; [
       glib # libglib-2.0.so.0, libgthread-2.0.so.0
       libGL
       clang
     ]
   );
};

X11_X11_INCLUDE_PATH = $(nix eval --raw --expr 'let pkgs = import <nixpkgs> {}; in "${pkgs.xorg.libX11}/lib"' --impure)