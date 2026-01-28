# Unity Implementation Guide

This document provides step-by-step instructions for implementing the Unity VR client that integrates with the Multi-Dimensional AI Creature backend.

## Overview

The Unity client is responsible for:

- Capturing sensory data from the VR environment (vision, audio, touch, proprioception)
- Sending this data to the Python backend via TCP
- Receiving actuator commands from the AI creature
- Applying these commands to the creature's avatar in real-time

## Prerequisites

### Required Software

- **Unity 2022.3 LTS or newer**
- **C# .NET 6.0+**
- **VR Headset SDK** (e.g., Meta Quest, SteamVR, Pico)
- **Unity Packages**:
    - XR Interaction Toolkit
    - TextMeshPro
    - Newtonsoft Json (for JSON serialization)

### Hardware Requirements

- **VR Headset**: Meta Quest 2/3/Pro, Valve Index, or compatible
- **Haptic Gloves**: Optional but recommended for touch data
- **External Microphone**: For audio capture
- **Network**: Low-latency local network connection to Python server

---

## Implementation Phases

### Phase 1: Project Setup

#### Step 1.1: Create Unity Project

1. Open Unity Hub
2. Click **New Project**
3. Select **3D (URP)** template for best VR performance
4. Name: `MultiDimensionalAI-VR`
5. Location: Choose your preferred directory
6. Click **Create Project**

#### Step 1.2: Install Required Packages

1. Open **Window → Package Manager**
2. Click **+** → **Add package from git URL**
3. Add the following packages:

```
com.unity.xr.interaction.toolkit@2.5.0
com.unity.xr.management@4.4.0
```

4. For Newtonsoft Json:
    - Download from [Unity Asset Store](https://assetstore.unity.com/packages/tools/input-management/json-net-for-unity-11347) OR
    - Add via Package Manager: `com.unity.nuget.newtonsoft-json@3.2.1`

#### Step 1.3: Configure XR Settings

1. Go to **Edit → Project Settings → XR Plug-in Management**
2. Enable your VR platform:
    - **PC VR**: Enable **OpenXR** or **SteamVR**
    - **Quest**: Enable **Oculus** or **OpenXR**
3. Configure interaction profiles for your controllers
4. Set **Render Mode** to **Multi-Pass** for stereoscopic vision

---

### Phase 2: Project Structure

#### Step 2.1: Create Folder Structure

Create the following folder hierarchy in your Unity project:

```
Assets/
├── AICreature/
│   ├── Scripts/
│   │   ├── Networking/       # TCP client and protocol
│   │   ├── Sensors/          # Vision, audio, touch, proprio
│   │   ├── Actuators/        # Voice, body control
│   │   ├── Core/             # Main manager, data structures
│   │   └── Utils/            # Helpers, serialization
│   ├── Prefabs/
│   │   ├── CreatureAvatar.prefab
│   │   └── SensorRig.prefab
│   ├── Materials/
│   ├── Models/               # 3D creature model
│   └── Audio/
├── Scenes/
│   ├── MainVR.unity
│   └── TestEnvironment.unity
└── ThirdParty/
	└── NewtonsoftJson/
```

#### Step 2.2: Create Core Scripts

You'll create these scripts in the following steps. The main scripts are:

| Script                    | Purpose                      |
| ------------------------- | ---------------------------- |
| `AICreatureManager.cs`    | Central coordinator          |
| `TCPClient.cs`            | Network communication        |
| `VRInputMessage.cs`       | Data structure for inputs    |
| `VROutputMessage.cs`      | Data structure for outputs   |
| `VisionSensor.cs`         | Capture stereo camera frames |
| `AudioSensor.cs`          | Capture microphone audio     |
| `TouchSensor.cs`          | Capture haptic contacts      |
| `ProprioceptionSensor.cs` | Capture joint states         |
| `VocalizationActuator.cs` | Play voice audio             |
| `AnimationActuator.cs`    | Control body movements       |

---

### Phase 3: Network Communication

#### Step 3.1: Create Protocol Data Structures

Create `Assets/AICreature/Scripts/Core/VRInputMessage.cs`:

```csharp
using System;
using System.Collections.Generic;
using Newtonsoft.Json;

namespace AICreature.Core
{
	[Serializable]
	public class VRInputMessage
	{
		[JsonProperty("timestamp")]
		public double Timestamp { get; set; }

		[JsonProperty("vision_left")]
		public string VisionLeft { get; set; } = null;

		[JsonProperty("vision_right")]
		public string VisionRight { get; set; } = null;

		[JsonProperty("audio_samples")]
		public List<float> AudioSamples { get; set; } = new List<float>();

		[JsonProperty("touch_contacts")]
		public List<Dictionary<string, object>> TouchContacts { get; set; } = new List<Dictionary<string, object>>();

		[JsonProperty("joint_positions")]
		public List<Dictionary<string, float>> JointPositions { get; set; } = new List<Dictionary<string, float>>();

		[JsonProperty("joint_rotations")]
		public List<Dictionary<string, float>> JointRotations { get; set; } = new List<Dictionary<string, float>>();

		public string ToJson()
		{
			return JsonConvert.SerializeObject(this);
		}

		public static VRInputMessage FromJson(string json)
		{
			return JsonConvert.DeserializeObject<VRInputMessage>(json);
		}
	}
}
```

Create `Assets/AICreature/Scripts/Core/VROutputMessage.cs`:

```csharp
using System;
using System.Collections.Generic;
using Newtonsoft.Json;

namespace AICreature.Core
{
	[Serializable]
	public class VROutputMessage
	{
		[JsonProperty("timestamp")]
		public double Timestamp { get; set; }

		[JsonProperty("vocalization_tokens")]
		public List<int> VocalizationTokens { get; set; } = new List<int>();

		[JsonProperty("vocalization_audio")]
		public string VocalizationAudio { get; set; } = null;

		[JsonProperty("joint_rotations")]
		public List<Dictionary<string, float>> JointRotations { get; set; } = new List<Dictionary<string, float>>();

		[JsonProperty("blend_shapes")]
		public List<Dictionary<string, float>> BlendShapes { get; set; } = new List<Dictionary<string, float>>();

		[JsonProperty("eye_params")]
		public Dictionary<string, float> EyeParams { get; set; } = new Dictionary<string, float>();

		public string ToJson()
		{
			return JsonConvert.SerializeObject(this);
		}

		public static VROutputMessage FromJson(string json)
		{
			return JsonConvert.DeserializeObject<VROutputMessage>(json);
		}
	}
}
```

#### Step 3.2: Create TCP Client

Create `Assets/AICreature/Scripts/Networking/TCPClient.cs`:

```csharp
using System;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

namespace AICreature.Networking
{
	public class TCPClient : MonoBehaviour
	{
		[Header("Connection Settings")]
		[SerializeField] private string serverHost = "localhost";
		[SerializeField] private int serverPort = 5555;
		[SerializeField] private float reconnectInterval = 5.0f;

		private TcpClient client;
		private NetworkStream stream;
		private bool isConnected = false;
		private float lastReconnectAttempt = 0f;
		private byte[] receiveBuffer = new byte[65536]; // 64KB buffer
		private int bufferPosition = 0;

		public bool IsConnected => isConnected;

		public event Action OnConnected;
		public event Action OnDisconnected;

		private void Update()
		{
			// Auto-reconnect if disconnected
			if (!isConnected && Time.time - lastReconnectAttempt > reconnectInterval)
			{
				TryConnect();
			}
		}

		public void TryConnect()
		{
			try
			{
				lastReconnectAttempt = Time.time;

				client = new TcpClient();
				client.Connect(serverHost, serverPort);
				stream = client.GetStream();
				isConnected = true;

				Debug.Log($"Connected to {serverHost}:{serverPort}");
				OnConnected?.Invoke();
			}
			catch (Exception e)
			{
				Debug.LogWarning($"Connection failed: {e.Message}");
				isConnected = false;
			}
		}

		public void Disconnect()
		{
			if (stream != null)
			{
				stream.Close();
				stream = null;
			}

			if (client != null)
			{
				client.Close();
				client = null;
			}

			isConnected = false;
			bufferPosition = 0;

			Debug.Log("Disconnected from server");
			OnDisconnected?.Invoke();
		}

		public bool SendMessage(string jsonMessage)
		{
			if (!isConnected || stream == null)
			{
				Debug.LogError("Cannot send message: Not connected");
				return false;
			}

			try
			{
				byte[] messageBytes = Encoding.UTF8.GetBytes(jsonMessage);

				// Frame message: 4-byte length prefix (big-endian) + message
				byte[] lengthPrefix = BitConverter.GetBytes((uint)messageBytes.Length);
				if (BitConverter.IsLittleEndian)
				{
					Array.Reverse(lengthPrefix); // Convert to big-endian
				}

				stream.Write(lengthPrefix, 0, 4);
				stream.Write(messageBytes, 0, messageBytes.Length);
				stream.Flush();

				return true;
			}
			catch (Exception e)
			{
				Debug.LogError($"Send failed: {e.Message}");
				Disconnect();
				return false;
			}
		}

		public string ReceiveMessage()
		{
			if (!isConnected || stream == null || !stream.DataAvailable)
			{
				return null;
			}

			try
			{
				// Read available data into buffer
				int bytesRead = stream.Read(receiveBuffer, bufferPosition, receiveBuffer.Length - bufferPosition);

				if (bytesRead == 0)
				{
					// Connection closed
					Debug.LogWarning("Server closed connection");
					Disconnect();
					return null;
				}

				bufferPosition += bytesRead;

				// Check if we have a complete message
				if (bufferPosition < 4)
				{
					return null; // Not enough data for length prefix
				}

				// Read length prefix (big-endian)
				byte[] lengthBytes = new byte[4];
				Array.Copy(receiveBuffer, 0, lengthBytes, 0, 4);

				if (BitConverter.IsLittleEndian)
				{
					Array.Reverse(lengthBytes);
				}

				uint messageLength = BitConverter.ToUInt32(lengthBytes, 0);

				// Check if we have the full message
				if (bufferPosition < 4 + messageLength)
				{
					return null; // Incomplete message
				}

				// Extract message
				string message = Encoding.UTF8.GetString(receiveBuffer, 4, (int)messageLength);

				// Shift remaining buffer data
				int remainingBytes = bufferPosition - (4 + (int)messageLength);
				if (remainingBytes > 0)
				{
					Array.Copy(receiveBuffer, 4 + (int)messageLength, receiveBuffer, 0, remainingBytes);
				}
				bufferPosition = remainingBytes;

				return message;
			}
			catch (Exception e)
			{
				Debug.LogError($"Receive failed: {e.Message}");
				Disconnect();
				return null;
			}
		}

		private void OnDestroy()
		{
			Disconnect();
		}

		private void OnApplicationQuit()
		{
			Disconnect();
		}
	}
}
```

---

### Phase 4: Sensor Implementation

#### Step 4.1: Vision Sensor

Create `Assets/AICreature/Scripts/Sensors/VisionSensor.cs`:

```csharp
using System;
using UnityEngine;

namespace AICreature.Sensors
{
	public class VisionSensor : MonoBehaviour
	{
		[Header("Camera Settings")]
		[SerializeField] private Camera leftEyeCamera;
		[SerializeField] private Camera rightEyeCamera;
		[SerializeField] private int captureWidth = 224;
		[SerializeField] private int captureHeight = 224;
		[SerializeField] private TextureFormat textureFormat = TextureFormat.RGB24;

		[Header("Encoding")]
		[SerializeField] private bool useJPEG = true;
		[SerializeField] private int jpegQuality = 75;

		private RenderTexture leftRenderTexture;
		private RenderTexture rightRenderTexture;
		private Texture2D leftReadTexture;
		private Texture2D rightReadTexture;

		private void Awake()
		{
			// Create render textures
			leftRenderTexture = new RenderTexture(captureWidth, captureHeight, 24);
			leftReadTexture = new Texture2D(captureWidth, captureHeight, textureFormat, false);

			if (rightEyeCamera != null)
			{
				rightRenderTexture = new RenderTexture(captureWidth, captureHeight, 24);
				rightReadTexture = new Texture2D(captureWidth, captureHeight, textureFormat, false);
			}
		}

		public string CaptureLeftEye()
		{
			return CaptureCamera(leftEyeCamera, leftRenderTexture, leftReadTexture);
		}

		public string CaptureRightEye()
		{
			if (rightEyeCamera == null)
				return null;

			return CaptureCamera(rightEyeCamera, rightRenderTexture, rightReadTexture);
		}

		private string CaptureCamera(Camera cam, RenderTexture rt, Texture2D tex)
		{
			if (cam == null)
			{
				Debug.LogError("Camera is null");
				return null;
			}

			try
			{
				// Render camera to texture
				RenderTexture previousRT = RenderTexture.active;
				cam.targetTexture = rt;
				cam.Render();

				// Read pixels
				RenderTexture.active = rt;
				tex.ReadPixels(new Rect(0, 0, captureWidth, captureHeight), 0, 0);
				tex.Apply();

				// Restore
				cam.targetTexture = null;
				RenderTexture.active = previousRT;

				// Encode to base64
				byte[] imageBytes;
				if (useJPEG)
				{
					imageBytes = tex.EncodeToJPG(jpegQuality);
				}
				else
				{
					imageBytes = tex.EncodeToPNG();
				}

				return Convert.ToBase64String(imageBytes);
			}
			catch (Exception e)
			{
				Debug.LogError($"Vision capture failed: {e.Message}");
				return null;
			}
		}

		private void OnDestroy()
		{
			if (leftRenderTexture != null)
				leftRenderTexture.Release();
			if (rightRenderTexture != null)
				rightRenderTexture.Release();

			Destroy(leftReadTexture);
			Destroy(rightReadTexture);
		}
	}
}
```

#### Step 4.2: Audio Sensor

Create `Assets/AICreature/Scripts/Sensors/AudioSensor.cs`:

```csharp
using System.Collections.Generic;
using UnityEngine;

namespace AICreature.Sensors
{
	public class AudioSensor : MonoBehaviour
	{
		[Header("Audio Settings")]
		[SerializeField] private int sampleRate = 48000;
		[SerializeField] private int bufferLength = 1024;
		[SerializeField] private bool useDefaultMicrophone = true;
		[SerializeField] private string specificMicrophone = "";

		private AudioClip microphoneClip;
		private int lastSamplePosition = 0;
		private List<float> audioBuffer = new List<float>();

		private void Start()
		{
			string micDevice = useDefaultMicrophone ? null : specificMicrophone;

			if (Microphone.devices.Length == 0)
			{
				Debug.LogError("No microphone detected!");
				return;
			}

			// Start recording
			microphoneClip = Microphone.Start(micDevice, true, 1, sampleRate);

			Debug.Log($"Audio sensor started with device: {micDevice ?? "Default"}");
		}

		public List<float> GetAudioSamples()
		{
			if (microphoneClip == null)
			{
				Debug.LogWarning("Microphone not initialized");
				return new List<float>();
			}

			int currentPosition = Microphone.GetPosition(null);

			if (currentPosition < lastSamplePosition)
			{
				// Wrapped around
				int samplesToEnd = microphoneClip.samples - lastSamplePosition;
				float[] samples = new float[samplesToEnd + currentPosition];

				microphoneClip.GetData(samples, lastSamplePosition);

				audioBuffer.Clear();
				audioBuffer.AddRange(samples);
			}
			else
			{
				// Normal case
				int sampleCount = currentPosition - lastSamplePosition;

				if (sampleCount > 0)
				{
					float[] samples = new float[sampleCount];
					microphoneClip.GetData(samples, lastSamplePosition);

					audioBuffer.Clear();
					audioBuffer.AddRange(samples);
				}
				else
				{
					audioBuffer.Clear();
				}
			}

			lastSamplePosition = currentPosition;
			return new List<float>(audioBuffer);
		}

		private void OnDestroy()
		{
			if (Microphone.IsRecording(null))
			{
				Microphone.End(null);
			}
		}
	}
}
```

#### Step 4.3: Touch Sensor

Create `Assets/AICreature/Scripts/Sensors/TouchSensor.cs`:

```csharp
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR;

namespace AICreature.Sensors
{
	public class TouchSensor : MonoBehaviour
	{
		[Header("Touch Settings")]
		[SerializeField] private float maxPressure = 1.0f;

		private InputDevice leftHand;
		private InputDevice rightHand;
		private bool devicesInitialized = false;

		private void Update()
		{
			if (!devicesInitialized)
			{
				InitializeDevices();
			}
		}

		private void InitializeDevices()
		{
			var leftHandDevices = new List<InputDevice>();
			var rightHandDevices = new List<InputDevice>();

			InputDevices.GetDevicesAtXRNode(XRNode.LeftHand, leftHandDevices);
			InputDevices.GetDevicesAtXRNode(XRNode.RightHand, rightHandDevices);

			if (leftHandDevices.Count > 0)
				leftHand = leftHandDevices[0];

			if (rightHandDevices.Count > 0)
				rightHand = rightHandDevices[0];

			devicesInitialized = (leftHandDevices.Count > 0 || rightHandDevices.Count > 0);
		}

		public List<Dictionary<string, object>> GetTouchContacts()
		{
			var contacts = new List<Dictionary<string, object>>();

			// Get grip/trigger values as proxy for touch
			if (leftHand.isValid)
			{
				leftHand.TryGetFeatureValue(CommonUsages.grip, out float leftGrip);
				leftHand.TryGetFeatureValue(CommonUsages.trigger, out float leftTrigger);

				contacts.Add(new Dictionary<string, object>
				{
					{ "hand", "left" },
					{ "grip", Mathf.Clamp01(leftGrip) },
					{ "trigger", Mathf.Clamp01(leftTrigger) }
				});
			}

			if (rightHand.isValid)
			{
				rightHand.TryGetFeatureValue(CommonUsages.grip, out float rightGrip);
				rightHand.TryGetFeatureValue(CommonUsages.trigger, out float rightTrigger);

				contacts.Add(new Dictionary<string, object>
				{
					{ "hand", "right" },
					{ "grip", Mathf.Clamp01(rightGrip) },
					{ "trigger", Mathf.Clamp01(rightTrigger) }
				});
			}

			return contacts;
		}
	}
}
```

#### Step 4.4: Proprioception Sensor

Create `Assets/AICreature/Scripts/Sensors/ProprioceptionSensor.cs`:

```csharp
using System.Collections.Generic;
using UnityEngine;

namespace AICreature.Sensors
{
	public class ProprioceptionSensor : MonoBehaviour
	{
		[Header("Joint Tracking")]
		[SerializeField] private List<Transform> trackedJoints = new List<Transform>();

		private Dictionary<Transform, Quaternion> previousRotations = new Dictionary<Transform, Quaternion>();
		private Dictionary<Transform, Vector3> previousPositions = new Dictionary<Transform, Vector3>();

		private void Start()
		{
			// Initialize previous states
			foreach (var joint in trackedJoints)
			{
				if (joint != null)
				{
					previousRotations[joint] = joint.localRotation;
					previousPositions[joint] = joint.localPosition;
				}
			}
		}

		public List<Dictionary<string, float>> GetJointPositions()
		{
			var positions = new List<Dictionary<string, float>>();

			foreach (var joint in trackedJoints)
			{
				if (joint == null)
					continue;

				Vector3 pos = joint.localPosition;
				positions.Add(new Dictionary<string, float>
				{
					{ "x", pos.x },
					{ "y", pos.y },
					{ "z", pos.z }
				});

				previousPositions[joint] = pos;
			}

			return positions;
		}

		public List<Dictionary<string, float>> GetJointRotations()
		{
			var rotations = new List<Dictionary<string, float>>();

			foreach (var joint in trackedJoints)
			{
				if (joint == null)
					continue;

				Vector3 euler = joint.localEulerAngles;
				rotations.Add(new Dictionary<string, float>
				{
					{ "x", euler.x },
					{ "y", euler.y },
					{ "z", euler.z }
				});

				previousRotations[joint] = joint.localRotation;
			}

			return rotations;
		}

		public void SetTrackedJoints(List<Transform> joints)
		{
			trackedJoints = joints;

			// Reinitialize tracking
			previousRotations.Clear();
			previousPositions.Clear();

			foreach (var joint in trackedJoints)
			{
				if (joint != null)
				{
					previousRotations[joint] = joint.localRotation;
					previousPositions[joint] = joint.localPosition;
				}
			}
		}
	}
}
```

---

### Phase 5: Actuator Implementation

#### Step 5.1: Vocalization Actuator

Create `Assets/AICreature/Scripts/Actuators/VocalizationActuator.cs`:

```csharp
using System;
using System.Collections;
using UnityEngine;

namespace AICreature.Actuators
{
	[RequireComponent(typeof(AudioSource))]
	public class VocalizationActuator : MonoBehaviour
	{
		[Header("Audio Settings")]
		[SerializeField] private int sampleRate = 48000;
		[SerializeField] private AudioSource audioSource;

		private void Awake()
		{
			if (audioSource == null)
			{
				audioSource = GetComponent<AudioSource>();
			}
		}

		public void PlayVocalization(string base64Audio)
		{
			if (string.IsNullOrEmpty(base64Audio))
			{
				Debug.LogWarning("Received empty vocalization");
				return;
			}

			try
			{
				byte[] audioBytes = Convert.FromBase64String(base64Audio);

				// Assuming audio is PCM float samples
				float[] samples = new float[audioBytes.Length / 4];
				Buffer.BlockCopy(audioBytes, 0, samples, 0, audioBytes.Length);

				// Create AudioClip
				AudioClip clip = AudioClip.Create("Vocalization", samples.Length, 1, sampleRate, false);
				clip.SetData(samples, 0);

				// Play
				audioSource.clip = clip;
				audioSource.Play();
			}
			catch (Exception e)
			{
				Debug.LogError($"Failed to play vocalization: {e.Message}");
			}
		}

		public void StopVocalization()
		{
			audioSource.Stop();
		}
	}
}
```

#### Step 5.2: Animation Actuator

Create `Assets/AICreature/Scripts/Actuators/AnimationActuator.cs`:

```csharp
using System.Collections.Generic;
using UnityEngine;

namespace AICreature.Actuators
{
	public class AnimationActuator : MonoBehaviour
	{
		[Header("Avatar Settings")]
		[SerializeField] private SkinnedMeshRenderer skinnedMeshRenderer;
		[SerializeField] private List<Transform> controllableJoints = new List<Transform>();

		[Header("Blend Settings")]
		[SerializeField] private float rotationSpeed = 10f;
		[SerializeField] private float blendSpeed = 5f;

		private Dictionary<string, int> blendShapeIndices = new Dictionary<string, int>();

		private void Start()
		{
			// Build blend shape index mapping
			if (skinnedMeshRenderer != null)
			{
				for (int i = 0; i < skinnedMeshRenderer.sharedMesh.blendShapeCount; i++)
				{
					string name = skinnedMeshRenderer.sharedMesh.GetBlendShapeName(i);
					blendShapeIndices[name] = i;
				}
			}
		}

		public void SetJointRotations(List<Dictionary<string, float>> rotations)
		{
			if (rotations == null || rotations.Count == 0)
				return;

			int count = Mathf.Min(rotations.Count, controllableJoints.Count);

			for (int i = 0; i < count; i++)
			{
				if (controllableJoints[i] == null)
					continue;

				var rot = rotations[i];

				if (rot.TryGetValue("x", out float x) &&
					rot.TryGetValue("y", out float y) &&
					rot.TryGetValue("z", out float z))
				{
					Quaternion targetRotation = Quaternion.Euler(x, y, z);
					controllableJoints[i].localRotation = Quaternion.Slerp(
						controllableJoints[i].localRotation,
						targetRotation,
						Time.deltaTime * rotationSpeed
					);
				}
			}
		}

		public void SetBlendShapes(List<Dictionary<string, float>> blendShapes)
		{
			if (blendShapes == null || blendShapes.Count == 0 || skinnedMeshRenderer == null)
				return;

			foreach (var blendShape in blendShapes)
			{
				foreach (var kvp in blendShape)
				{
					if (blendShapeIndices.TryGetValue(kvp.Key, out int index))
					{
						float currentWeight = skinnedMeshRenderer.GetBlendShapeWeight(index);
						float targetWeight = kvp.Value * 100f; // 0-100 range
						float newWeight = Mathf.Lerp(currentWeight, targetWeight, Time.deltaTime * blendSpeed);

						skinnedMeshRenderer.SetBlendShapeWeight(index, newWeight);
					}
				}
			}
		}

		public void SetEyeParameters(Dictionary<string, float> eyeParams)
		{
			// Implement eye movement logic here
			// This depends on your avatar's eye setup

			if (eyeParams == null || eyeParams.Count == 0)
				return;

			// Example: Set gaze direction
			if (eyeParams.TryGetValue("gaze_x", out float gazeX) &&
				eyeParams.TryGetValue("gaze_y", out float gazeY))
			{
				// Apply to eye transforms
				// This is a placeholder - implement based on your avatar
			}
		}

		public void SetControllableJoints(List<Transform> joints)
		{
			controllableJoints = joints;
		}
	}
}
```

---

### Phase 6: Main Manager

#### Step 6.1: Create AI Creature Manager

Create `Assets/AICreature/Scripts/Core/AICreatureManager.cs`:

```csharp
using System;
using UnityEngine;
using AICreature.Core;
using AICreature.Networking;
using AICreature.Sensors;
using AICreature.Actuators;

namespace AICreature.Core
{
	public class AICreatureManager : MonoBehaviour
	{
		[Header("Network")]
		[SerializeField] private TCPClient tcpClient;

		[Header("Sensors")]
		[SerializeField] private VisionSensor visionSensor;
		[SerializeField] private AudioSensor audioSensor;
		[SerializeField] private TouchSensor touchSensor;
		[SerializeField] private ProprioceptionSensor proprioceptionSensor;

		[Header("Actuators")]
		[SerializeField] private VocalizationActuator vocalizationActuator;
		[SerializeField] private AnimationActuator animationActuator;

		[Header("Settings")]
		[SerializeField] private float updateRate = 30f; // 30 Hz
		[SerializeField] private bool captureRightEye = false;

		private float lastUpdateTime = 0f;
		private double sessionStartTime;

		private void Start()
		{
			sessionStartTime = Time.realtimeSinceStartupAsDouble;

			// Connect to server
			if (tcpClient == null)
			{
				tcpClient = GetComponent<TCPClient>();
			}

			if (tcpClient != null)
			{
				tcpClient.TryConnect();
			}
			else
			{
				Debug.LogError("TCPClient not found!");
			}
		}

		private void Update()
		{
			if (tcpClient == null || !tcpClient.IsConnected)
				return;

			// Check if it's time to send update
			float timeSinceLastUpdate = Time.time - lastUpdateTime;
			float updateInterval = 1f / updateRate;

			if (timeSinceLastUpdate < updateInterval)
				return;

			lastUpdateTime = Time.time;

			// Collect sensor data
			VRInputMessage inputMsg = CollectSensorData();

			// Send to server
			string inputJson = inputMsg.ToJson();
			bool sent = tcpClient.SendMessage(inputJson);

			if (!sent)
			{
				Debug.LogWarning("Failed to send input message");
				return;
			}

			// Receive response
			string outputJson = tcpClient.ReceiveMessage();

			if (!string.IsNullOrEmpty(outputJson))
			{
				try
				{
					VROutputMessage outputMsg = VROutputMessage.FromJson(outputJson);
					ApplyActuatorCommands(outputMsg);
				}
				catch (Exception e)
				{
					Debug.LogError($"Failed to parse output message: {e.Message}");
				}
			}
		}

		private VRInputMessage CollectSensorData()
		{
			double timestamp = Time.realtimeSinceStartupAsDouble - sessionStartTime;

			VRInputMessage msg = new VRInputMessage
			{
				Timestamp = timestamp
			};

			// Vision
			if (visionSensor != null)
			{
				msg.VisionLeft = visionSensor.CaptureLeftEye();

				if (captureRightEye)
				{
					msg.VisionRight = visionSensor.CaptureRightEye();
				}
			}

			// Audio
			if (audioSensor != null)
			{
				msg.AudioSamples = audioSensor.GetAudioSamples();
			}

			// Touch
			if (touchSensor != null)
			{
				msg.TouchContacts = touchSensor.GetTouchContacts();
			}

			// Proprioception
			if (proprioceptionSensor != null)
			{
				msg.JointPositions = proprioceptionSensor.GetJointPositions();
				msg.JointRotations = proprioceptionSensor.GetJointRotations();
			}

			return msg;
		}

		private void ApplyActuatorCommands(VROutputMessage msg)
		{
			// Vocalization
			if (vocalizationActuator != null && !string.IsNullOrEmpty(msg.VocalizationAudio))
			{
				vocalizationActuator.PlayVocalization(msg.VocalizationAudio);
			}

			// Animation
			if (animationActuator != null)
			{
				if (msg.JointRotations != null && msg.JointRotations.Count > 0)
				{
					animationActuator.SetJointRotations(msg.JointRotations);
				}

				if (msg.BlendShapes != null && msg.BlendShapes.Count > 0)
				{
					animationActuator.SetBlendShapes(msg.BlendShapes);
				}

				if (msg.EyeParams != null && msg.EyeParams.Count > 0)
				{
					animationActuator.SetEyeParameters(msg.EyeParams);
				}
			}
		}

		private void OnDestroy()
		{
			if (tcpClient != null && tcpClient.IsConnected)
			{
				tcpClient.Disconnect();
			}
		}
	}
}
```

---

### Phase 7: Scene Setup

#### Step 7.1: Create Main Scene

1. Open Unity and create a new scene: **File → New Scene → VR**
2. Save as `Assets/Scenes/MainVR.unity`

#### Step 7.2: Setup XR Rig

1. In Hierarchy, right-click → **XR → XR Origin (Action-based)**
2. This creates:
    - `XR Origin` (parent)
    - `Camera Offset`
    - `Main Camera` (left eye)
    - `Left Controller`
    - `Right Controller`

#### Step 7.3: Add Creature Avatar

1. Import your 3D creature model into `Assets/AICreature/Models/`
2. Drag the model into the scene
3. Position it in front of the camera (e.g., 2 units away)
4. Name it `CreatureAvatar`
5. Ensure it has:
    - `SkinnedMeshRenderer` component
    - Rigged skeleton with named bones
    - Blend shapes for facial expressions

#### Step 7.4: Setup AI Manager

1. Create empty GameObject: **Hierarchy → Create Empty**
2. Name it `AICreatureManager`
3. Add component: `AICreatureManager.cs`
4. Add component: `TCPClient.cs`

#### Step 7.5: Setup Sensors

**Vision Sensor:**

1. Select `XR Origin/Camera Offset/Main Camera`
2. Add component: `VisionSensor.cs`
3. In Inspector:
    - Set **Left Eye Camera** to `Main Camera`
    - **Capture Width/Height**: 224x224
    - **Use JPEG**: true
    - **JPEG Quality**: 75

**Audio Sensor:**

1. Select `AICreatureManager`
2. Add component: `AudioSensor.cs`
3. In Inspector:
    - **Sample Rate**: 48000
    - **Buffer Length**: 1024
    - **Use Default Microphone**: true

**Touch Sensor:**

1. Select `AICreatureManager`
2. Add component: `TouchSensor.cs`

**Proprioception Sensor:**

1. Select `AICreatureManager`
2. Add component: `ProprioceptionSensor.cs`
3. In Inspector, add tracked joints:
    - Expand `CreatureAvatar` skeleton
    - Drag important bones to **Tracked Joints** list
    - Example: spine, neck, head, arms, hands

#### Step 7.6: Setup Actuators

**Vocalization Actuator:**

1. Select `CreatureAvatar`
2. Add component: `Audio Source`
3. Add component: `VocalizationActuator.cs`
4. In Inspector:
    - **Sample Rate**: 48000
    - **Audio Source**: Auto-populated

**Animation Actuator:**

1. Select `CreatureAvatar`
2. Add component: `AnimationActuator.cs`
3. In Inspector:
    - **Skinned Mesh Renderer**: Drag the creature's renderer
    - **Controllable Joints**: Add joints you want AI to control
    - **Rotation Speed**: 10
    - **Blend Speed**: 5

#### Step 7.7: Wire Up Manager

1. Select `AICreatureManager`
2. In Inspector, populate references:
    - **TCP Client**: Drag `AICreatureManager` (self)
    - **Vision Sensor**: Drag `Main Camera` (which has VisionSensor)
    - **Audio Sensor**: Drag `AICreatureManager` (self)
    - **Touch Sensor**: Drag `AICreatureManager` (self)
    - **Proprioception Sensor**: Drag `AICreatureManager` (self)
    - **Vocalization Actuator**: Drag `CreatureAvatar`
    - **Animation Actuator**: Drag `CreatureAvatar`
    - **Update Rate**: 30
    - **Capture Right Eye**: false (optional)

---

### Phase 8: Testing

#### Step 8.1: Test Network Connection

1. Start the Python server:

```bash
python scripts/vr_server.py --host localhost --port 5555
```

2. In Unity, click **Play**
3. Check Console for "Connected to localhost:5555"

#### Step 8.2: Test Sensors

Add debug logging to verify data collection:

1. Open `AICreatureManager.cs`
2. In `CollectSensorData()`, add after collecting each sensor:

```csharp
Debug.Log($"Vision: {msg.VisionLeft != null}");
Debug.Log($"Audio samples: {msg.AudioSamples.Count}");
Debug.Log($"Touch contacts: {msg.TouchContacts.Count}");
```

3. Click **Play** and check Console output

#### Step 8.3: Test Actuators

Send test commands from Python:

```python
from src.vr_integration.protocol import VROutputMessage

# Create test message
msg = VROutputMessage(
	timestamp=0.0,
	joint_rotations=[
		{"x": 10, "y": 0, "z": 0},
		{"x": 0, "y": 15, "z": 0}
	],
	blend_shapes=[
		{"smile": 0.8}
	]
)

# Send to Unity
server.send_message(msg)
```

4. Verify creature avatar animates in Unity

---

### Phase 9: Optimization

#### Step 9.1: Performance Profiling

1. Open **Window → Analysis → Profiler**
2. Run scene and monitor:
    - **CPU Usage**: Should be < 16ms for 60 FPS
    - **Rendering**: GPU time
    - **Memory**: Watch for leaks
    - **Network**: Bandwidth usage

#### Step 9.2: Optimize Vision Capture

If vision capture is slow:

1. Reduce resolution: 224x224 → 128x128
2. Lower JPEG quality: 75 → 60
3. Capture every other frame:

```csharp
private int frameCounter = 0;

if (frameCounter % 2 == 0)
{
	msg.VisionLeft = visionSensor.CaptureLeftEye();
}
frameCounter++;
```

#### Step 9.3: Optimize Network

Reduce message frequency if needed:

1. In `AICreatureManager`, set **Update Rate** to 20 or 15 Hz
2. Use UDP instead of TCP for real-time data (optional advanced modification)

---

### Phase 10: Data Recording

#### Step 10.1: Enable Recording Mode

To collect training data:

1. In Python, start the VR server with recorder:

```bash
python scripts/record_vr_session.py --output data_root/session_001
```

2. In Unity, play the scene normally
3. Data will be automatically recorded to disk

#### Step 10.2: Verify Recorded Data

After recording:

```bash
python scripts/validate_session.py data_root/session_001
```

Check for:

- ✅ All files present
- ✅ Timestamp continuity
- ✅ Sensor ranges valid

---

## Troubleshooting

### Connection Issues

| Issue                    | Solution                                        |
| ------------------------ | ----------------------------------------------- |
| **"Connection refused"** | Ensure Python server is running first           |
| **Timeout on connect**   | Check firewall settings, use `localhost` not IP |
| **Disconnects randomly** | Increase buffer size in `TCPClient.cs`          |

### Sensor Issues

| Issue                  | Solution                                         |
| ---------------------- | ------------------------------------------------ |
| **No vision data**     | Check camera render textures are created         |
| **No audio**           | Grant microphone permissions in Unity            |
| **Touch always zero**  | Verify VR controllers are paired                 |
| **Joints not tracked** | Ensure joints are added to `Tracked Joints` list |

### Performance Issues

| Issue             | Solution                                               |
| ----------------- | ------------------------------------------------------ |
| **Low FPS in VR** | Lower vision resolution, reduce update rate            |
| **High latency**  | Use wired connection, reduce message size              |
| **Memory leak**   | Check textures are destroyed properly in `OnDestroy()` |

---

## Advanced Features

### Multi-Creature Support

To support multiple AI creatures in one scene:

1. Modify `TCPClient` to accept unique creature ID
2. Start multiple Python servers on different ports
3. Each creature connects to its own server

### Haptic Feedback

To add haptic gloves:

1. Integrate glove SDK (e.g., SenseGlove, HaptX)
2. Extend `TouchSensor` to read per-finger pressure
3. Add haptic output to `VROutputMessage`

### Voice Chat Integration

To enable voice interaction:

1. Add speech-to-text in Unity (e.g., Azure Speech SDK)
2. Send transcribed text in `VRInputMessage`
3. Receive TTS audio in `VROutputMessage`

---

## Next Steps

After completing Unity implementation:

1. **Test with trained model**: Load checkpoint and run inference
2. **Collect training data**: Record diverse VR sessions
3. **Iterate on avatar**: Improve creature model and animations
4. **Scale up**: Test with 1B+ parameter models
5. **Deploy**: Build standalone VR application

---

## Resources

### Unity Learning

- [XR Interaction Toolkit Documentation](https://docs.unity3d.com/Packages/com.unity.xr.interaction.toolkit@2.5/manual/index.html)
- [VR Best Practices](https://docs.unity3d.com/Manual/VROverview.html)

### Networking

- [C# TCP Client Tutorial](https://learn.microsoft.com/en-us/dotnet/api/system.net.sockets.tcpclient)
- [Unity Networking Examples](https://github.com/Unity-Technologies/multiplayer-community-contributions)

### Related Documentation

- [VR Data Collection Process](vr_data_collection.md)
- [Dataset Formats](dataset_formats.md)
- [Training Guide](training_guide.md)

---

## Appendix: Complete File Checklist

Before testing, ensure you have created all files:

**Scripts:**

- ✅ `VRInputMessage.cs`
- ✅ `VROutputMessage.cs`
- ✅ `TCPClient.cs`
- ✅ `VisionSensor.cs`
- ✅ `AudioSensor.cs`
- ✅ `TouchSensor.cs`
- ✅ `ProprioceptionSensor.cs`
- ✅ `VocalizationActuator.cs`
- ✅ `AnimationActuator.cs`
- ✅ `AICreatureManager.cs`

**Scene Setup:**

- ✅ XR Origin configured
- ✅ Creature avatar with skinned mesh
- ✅ All sensors attached
- ✅ All actuators attached
- ✅ Manager references wired

**External:**

- ✅ Python server running
- ✅ VR headset connected
- ✅ Microphone enabled
