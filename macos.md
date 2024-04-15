# Compilation on macOS (macOS 14 on M-Series Macs)

## Installing dependencies

The following dependencies are required for compilation:
- Xcode Command Line Tools: `xcode-select --install`
- [Homebrew](https://brew.sh/): `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
- gcc / g++: `brew install gcc`
- .NET 6.0: [Install .NET on macOS](https://learn.microsoft.com/en-us/dotnet/core/install/macos)

## Building the matrix multiplication backend (C++)

To build the C++ library used for matrix multiplication the following steps are
required:
- Navigate to the `VectorSearch` directory: `cd VectorSearch`
- Build the DLL: `g++-13 -shared -I eigen-3.4.0 -fPIC -fopenmp -O3 -o VectorSearch.dll dllmainUnix.cpp`
- In case you get an error from the linker, you might want to try this instead: `g++-13 -Wl,-ld_classic -shared -I eigen-3.4.0 -fPIC -fopenmp -O3 -o VectorSearch.dll dllmainUnix.cpp`

## Building the prototype testing suite (C#)

To build the C# testing application the following steps need to be carried out:
- Navigate to the `DataLoader` directory: `cd DataLoader`
- Build the executable: `dotnet publish DataLoader.csproj --runtime osx-arm64 --self-contained --configuration Release`

## Running the executable

- Copy `VectorSearch.dll` to the build directory of `DataLoader`.
- Run the executable. For example with `./DataLoader EigenSIntB 10000`.
- Note that we only supply the compiled binaries for CPU-based search. GPU-based search is not available because Macs don't support Nvidia GPUs!

# Running the executable without building / compiling yourself

We provide compiled binaries in `DataLoaderExecutable/macos_arm` and in the
`Releases` tab. Please again note that we only supply the compiled binaries for
CPU-based search. Running the compiled binaries requires [.NET 6.0](https://learn.microsoft.com/en-us/dotnet/core/install/macos)
and [g++](#Installing-dependencies).
