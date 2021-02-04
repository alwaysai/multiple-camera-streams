# Multiple Camera Streams

This alwaysAI application demonstrates how to incorporate video streams from multiple cameras into a single app.

## Setup
This app requires an alwaysAI account. Head to the [Sign up page](https://www.alwaysai.co/dashboard) if you don't have an account yet. Follow the instructions to install the alwaysAI toolchain on your development machine.

Next, create an empty project to be used with this app. When you clone this repo, you can run `aai app configure` within the repo directory and your new project will appear in the list.

## Usage
Once you have the alwaysAI tools installed and the new project created, run the following CLI commands at the top level of the repo:

To set the project, and select the target device run:

```
aai app configure
```

To build your app and install on the target device:

```
aai app install
```

To start the app:

```
aai app start
```

## Design
This app shows how to perform inferencing on multiple camera streams in an asynchronous manner. That means that if one of the cameras is slower or gets disconnected, the main app will continue to run and gather results from the other cameras at the same rate. This app uses [WebcamVideoStream](https://alwaysai.co/docs/edgeiq_api/video_stream.html#edgeiq.edge_tools.WebcamVideoStream) for the camera video streams, which is useful for cameras directly connected to your device. For handling multiple IP streams, simply use [IPVideoStream](https://alwaysai.co/docs/edgeiq_api/video_stream.html#edgeiq.edge_tools.IPVideoStream) instead and change the indices to URLs.

An alternative approach is to read frames asynchrounously, but to perform inferencing in a synchronous way. This could be accomplished by making `CameraThread` send only the frame in the results object, and perform inferencing in `main()` once a frame has been grabbed from all cameras.
