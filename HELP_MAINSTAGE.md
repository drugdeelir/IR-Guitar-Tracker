# Interfacing with Apple MainStage over Ethernet

This guide explains how to control the Projection Mapper on Windows using MIDI from Apple MainStage on a Mac over an Ethernet connection.

## Requirements

1.  **Hardware:** Mac Mini (or any Mac) and a Windows PC connected via an Ethernet cable.
2.  **Software (Windows):** **rtpMIDI** (Free, download from [tobias-erichsen.de](https://www.tobias-erichsen.de/software/rtpmidi.html)). This adds the "Network MIDI" capability to Windows that is native to macOS.

---

## Step 1: Network Configuration

To ensure a stable connection, set static IP addresses for both machines.

1.  **On Mac:** Go to System Settings > Network > Ethernet > Details > TCP/IP. Set "Configure IPv4" to "Manually".
    - IP Address: `192.168.1.10`
    - Subnet Mask: `255.255.255.0`
2.  **On Windows:** Go to Control Panel > Network and Sharing Center > Change adapter settings. Right-click Ethernet > Properties > IPv4 > Properties.
    - IP Address: `192.168.1.100`
    - Subnet Mask: `255.255.255.0`
    - Gateway: `192.168.1.1`
    - DNS: `8.8.8.8` / `8.8.4.4`

---

## Step 2: Set up MIDI Network on Mac

1.  Open **Audio MIDI Setup** on your Mac (Found in Applications > Utilities).
2.  Go to **Window > Show MIDI Studio**.
3.  Double-click the **Network** icon.
4.  In the "My Sessions" panel, click **+** to create a new session. Call it "MainStageVJSync".
5.  Check the box to enable it.
6.  Set "Who can connect to me" to **Anyone**.

---

## Step 3: Set up MIDI Network on Windows

1.  Open **rtpMIDI** on your Windows PC.
2.  In the "My Sessions" panel, click **+** to create a new session. Call it "WindowsVJSink".
3.  Check the box to enable it.
4.  You should see your Mac's session ("MainStageVJSync") appear in the **Directory** panel.
5.  Select it and click **Connect**.
6.  The Mac session should now appear in the "Participants" list.

---

## Step 4: Configure MainStage

1.  Open your Concert in **MainStage**.
2.  Go to the **Layout** mode.
3.  Select the MIDI control you want to use (e.g., a button or knob on your Arturia keyboard).
4.  In the **Edit** mode, select the control and go to the **Mappings** tab.
5.  Set the control to send MIDI data out to the **Network Session** (MainStageVJSync).
6.  *Tip:* Use "External MIDI" channel strips to send MIDI Note or CC data directly to the network session on specific patch changes.

---

## Step 5: Map in Projection Mapper

1.  Launch **Projection Mapper** on Windows.
2.  In the **MIDI Settings** group, select "rtpMIDI - WindowsVJSink" from the **MIDI Input Port** dropdown.
3.  Click **Configure MIDI Mappings**.
4.  Click a button (e.g., "Toggle Amp Visibility"). It will say "Listening...".
5.  Trigger the corresponding control in MainStage or press the key on your Arturia keyboard.
6.  The mapping is now saved!

---

## Alternative: OSC Control

If you prefer OSC, the application listens on all network interfaces on port `8000`.
In your OSC-compatible software on the Mac, send messages to `192.168.1.100:8000`.

**Example OSC Paths:**
- `/mask/amp/visible [0 or 1]`
- `/mask/bg/fx/glitch [0 or 1]`
- `/style [acid, noir, retro, none]`
- `/snapshot [0-7]`
