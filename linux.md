# Compilation on Ubuntu 22.04 (and other Linux-based system)

This is a rough guide for compiling *CandidateVectorSearch* on Ubuntu, which should
generally work on most Linux-based systems.

## Installing dependencies

The following dependencies are required for compilation:
- gcc / g++: `sudo apt-get install -y gcc g++`
- dotnet \[[more info](https://learn.microsoft.com/en-us/dotnet/core/install/linux)\]:
  - dotnet-sdk-6.0: `sudo apt-get install -y dotnet-sdk-6.0`
  - dotnet-runtime-6.0: `sudo apt-get install -y dotnet-runtime-6.0`
- Additional dependencies may be required, please check `Dockerfile` to see a
  full list!

Alternatively, a [Docker](https://docs.docker.com/engine/install/) image with all necessary build tools is also available
via `Dockerfile`:
- Build the Dockerfile: `docker build . -f Dockerfile -t cvs`
- Run the container: `docker run -it cvs`

## Building the matrix multiplication backend (C++)

To build the C++ library used for matrix multiplication the following steps are
required:
- Navigate to the `VectorSearch` directory: `cd VectorSearch`
- Build the DLL: `g++ -shared -I eigen-3.4.0 -fPIC -fopenmp -O3 -o VectorSearch.dll dllmainUnix.cpp`

## Building the prototype testing suite (C#)

To build the C# testing application the following steps need to be carried out:
- Navigate to the `DataLoader` directory: `cd DataLoader`
- Build the executable: `dotnet publish DataLoader.csproj --runtime ubuntu.22.04-x64 --self-contained --configuration Release`
- \[Alternatively: `dotnet publish DataLoader.csproj --runtime linux-x64 --self-contained --configuration Release`\]

## Running the executable

- Copy `VectorSearch.dll` to the build directory of `DataLoader`.
- Run the executable. For example with `./DataLoader EigenSIntB 10000`.
- Note that we only supply the compiled binaries for CPU-based search. If you want
  to run GPU-based searches please compile `VectorSearchCUDA` yourself!

# Running the executable without building / compiling yourself

We provide compiled binaries in `DataLoaderExecutable/ubuntu22.04_x64` and in the
`Releases` tab. Please again note that we only supply the compiled binaries for
CPU-based search. If you want to run GPU-based searches please compile
`VectorSearchCUDA` yourself! Running the compiled binaries requires
`dotnet-runtime-6.0` and `g++`.
