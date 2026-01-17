# DMX/OSC Bridge Guide for Melodeus

A bridge that connects Melodeus voice events to stage lighting (DMX via Art-Net) and audio routing (Behringer X32 via OSC).

## Quick Start

```bash
cd /Users/olena/melodeus/voicetest
source venv/bin/activate
python dmx_osc_bridge.py
```

Dashboard opens at: **http://localhost:8090**

## Requirements

```bash
pip install python-osc pyartnet aiohttp
```

## Network Setup

| Device | IP Address | Protocol | Port |
|--------|------------|----------|------|
| eDMX1 Art-Net Node | 192.168.2.112 | Art-Net | 6454 |
| Mac (source) | 192.168.2.12 | - | - |
| Behringer X32 | 192.168.2.61 | OSC | 10023 |
| Melodeus OSC Input | 0.0.0.0 | OSC | 7000 |

## Architecture

```
┌─────────────┐     OSC      ┌──────────────┐     Art-Net    ┌─────────────┐
│  Melodeus   │────────────▶│   DMX/OSC    │──────────────▶│   eDMX1     │
│  (TTS/STT)  │  port 7000  │   Bridge     │   port 6454   │  (DMX out)  │
└─────────────┘              └──────────────┘                └──────────────┘
                                   │
                                   │ OSC
                                   ▼ port 10023
                              ┌──────────────┐
                              │  Behringer   │
                              │     X32      │
                              └──────────────┘
```

## Configuration File (`dmx_config.yaml`)

### Art-Net Settings

```yaml
artnet:
  enabled: true
  host: "192.168.2.112"      # eDMX1 IP
  source_ip: "192.168.2.12"  # Mac's ethernet IP
  port: 6454
  universe: 0
  fps: 40
```

### X32 Settings

```yaml
x32:
  enabled: true
  host: "192.168.2.61"
  port: 10023
```

### Character/Mannequin Configuration

Each character (AI mannequin) has:
- A **par can light** that brightens when speaking
- An **X32 bus** for individual speaker audio

```yaml
characters:
  Opus:  # Character name (must match melodeus config)
    par_can:
      dmx_address: 424        # Starting DMX address
      channels: 8             # 8-channel mode
      color: [255, 80, 50, 100, 180, 0]  # R, G, B, Lime, Amber, UV
      idle_brightness: 0.1    # 10% when not speaking
      active_brightness: 1.0  # 100% when speaking
    x32_channel: 1            # X32 Bus number
    x32_type: bus             # "bus", "ch", "auxin", or "fxrtn"
    x32_idle_fader: 0.0       # Fader when idle
    x32_active_fader: 0.75    # Fader when speaking
    x32_mute_idle: true       # Mute when not speaking
```

### Current Mannequin Assignments

| Character | DMX Address | X32 Bus | Color |
|-----------|-------------|---------|-------|
| Opus (3) | 424 | Bus 1 | Deep orange/amber |
| Opus4 | 432 | Bus 4 | Purple |
| Opus41 | 408 | Bus 3 | Purple |
| Opus45 | 400 | Bus 5 | Warm orange |
| Sonnet36 | 440 | Bus 6 | Green-ish |
| Sonnet4 | 448 | Bus 6 | Cyan/teal |
| Sonnet45 | 416 | Bus 2 | Cool blue |
| Sonnet35 | 456 | Bus 5 | Lavender |
| 3Sonnet | 464 | Bus 1 | Light blue |
| OpusAlt | 472 | Bus 1 | Red/orange |
| Haiku | - | Bus 2 | (no light) |
| Haiku45 | - | Bus 2 | (no light) |

### House Lights Configuration

Stage lights not tied to characters:

```yaml
house_lights:
  back_stage:
    - {name: "Back1", dmx_address: 1, type: "ADJ 7P HEX 13ch", channels: 13}
    # ... more fixtures
  front_stage:
    - {name: "FrontL", dmx_address: 110, type: "ADJ 7P HEX 13ch", channels: 13}
  front_spots:
    - {name: "FrontSpotL_200", dmx_address: 200, type: "Small Mover 15ch", channels: 15}
  big_movers:
    - {name: "BigMoverL_301", dmx_address: 301, type: "Chauvet Intimidator 375ZX 15ch", channels: 15}
```

### Mover Target Positions

Pre-programmed positions for big movers to aim at mannequins:

```yaml
mover_targets:
  Opus:  # Target character
    BigMoverL_301: {pan: 73, tilt: 237, zoom: 2}
    BigMoverR_316: {pan: 21, tilt: 233, zoom: 6}
```

## OSC Messages from Melodeus

The bridge listens for these OSC messages on port 7000:

| Address | Args | Description |
|---------|------|-------------|
| `/character/speaking/start` | character_name | Character started speaking |
| `/character/speaking/stop` | character_name | Character stopped speaking |
| `/character/thinking/start` | character_name | Character started thinking |
| `/character/thinking/stop` | character_name | Character stopped thinking |
| `/character/move_body` | char, body | Character possesses another body |
| `/avatar/color` | char, r, g, b | Change character's light color |
| `/test` | any | Test message (logged) |

## Web Dashboard

### Mannequin Controls

Each mannequin card shows:
- **Speaking indicator** - Glows when active
- **Color preview** - Current light color
- **Brightness slider** - Manual override (0-100%)
- **Color picker** - Change par can color
- **X32 fader** - Manual audio level
- **Activate/Deactivate** - Manual speaking state

### Global Controls

- **⬛ Blackout** - All lights off
- **🌙 All Idle** - All mannequins to idle state
- **☀️ All Full** - All mannequins full brightness
- **🔄 Refresh X32** - Re-send all X32 states
- **🔊 Master Mannequin Volume** - Scales all mannequin faders

### House Lights Section

Controls for stage lighting groups:
- **back_stage** - 7x ADJ 7P HEX on back wall
- **front_stage** - 2x ADJ 7P HEX downstage
- **front_spots** - 2x Small moving heads
- **big_movers** - 2x Chauvet Intimidator 375ZX

Each light has:
- Brightness slider
- Color picker
- Test button (2-second flash)

### Moving Heads Section

Controls for each mover:
- **Pan/Tilt sliders** - Position control (0-255)
- **Dimmer slider** - Brightness (0-100%)
- **Zoom slider** - For Intimidators only
- **Color buttons** - Color wheel presets
- **Quick buttons** - On/Off/Center
- **🎯 3Opus** - Move big movers to preset 3Opus position

## DMX Channel Maps

### ADJ Mega HEX Par / 7P HEX (8-channel)

| Channel | Function |
|---------|----------|
| 1 | Red |
| 2 | Green |
| 3 | Blue |
| 4 | White (7P) / Lime (Mega) |
| 5 | Amber |
| 6 | UV |
| 7 | Dimmer |
| 8 | Strobe (255=solid on) |

### ADJ 7P HEX (13-channel)

Same as 8-channel, plus:
| 9 | Color Macro |
| 10 | Auto Program |
| 11 | Speed |
| 12 | Fade |
| 13 | Dim Mode |

### Small Mover (15-channel)

| Channel | Function |
|---------|----------|
| 1 | Pan |
| 2 | Pan Fine |
| 3 | Tilt |
| 4 | Tilt Fine |
| 5 | Color Wheel |
| 6 | Gobo |
| 7 | Gobo Rotation |
| 8 | **Dimmer** |
| 9 | **Shutter** (255=open) |
| 10 | Focus |
| 11-15 | Prism/Programs |

**Notes:**
- Pan 0 = house front, Pan 96 = back
- Tilt 0 = angled back, Tilt 128 = straight down

### Chauvet Intimidator 375ZX (15-channel)

| Channel | Function |
|---------|----------|
| 1 | Pan |
| 2 | Pan Fine |
| 3 | Tilt |
| 4 | Tilt Fine |
| 5 | Pan/Tilt Speed |
| 6 | Color Wheel |
| 7 | Gobo |
| 8 | Gobo Rotation |
| 9 | Prism |
| 10 | Focus |
| 11 | **Dimmer** |
| 12 | **Shutter** (255=open) |
| 13 | Functions |
| 14 | Macros |
| 15 | **Zoom** |

**Notes:**
- Tilt 128 = up (pointed at ceiling)
- Has motorized zoom control

## X32 Routing Setup

The X32 is configured to route TTS audio to individual mannequin speakers:

1. **TTS Input**: Channels 17/18 (stereo from computer)
2. **Bus Routing**: Ch 17/18 sends to Buses 1-6 at unity
3. **Main Output**: Ch 17/18 → Main L/R with DELAY (for main speakers)
4. **Bus Outputs**: Bus 1-6 → Individual mannequin speakers (no delay)
5. **Mic Inputs**: Direct to Main L/R (no delay)

The bridge controls **Bus faders** to select which mannequin speaker plays.

## Troubleshooting

### Lights Not Responding

1. Check Art-Net IP (eDMX1 must be at 192.168.2.112)
2. Verify Mac is on correct network (192.168.2.x)
3. Check DMX addresses in config match physical fixture settings
4. Try the dashboard Test button to flash individual lights

### X32 Not Responding

1. Check X32 IP (should be 192.168.2.61)
2. Verify OSC is enabled on X32 (Setup → Network → OSC)
3. Use "Refresh X32" button to re-send states

### Melodeus Events Not Working

1. Check melodeus `config.yaml` has `osc.port: 7000`
2. Verify bridge shows "OSC server listening on 0.0.0.0:7000"
3. Watch the event log in the dashboard for incoming messages

### Movers Not Moving

1. Check DMX address matches fixture (Front spots: 200, 216; Big movers: 301, 316)
2. Verify channel mode (15-channel for all current movers)
3. Make sure dimmer is up AND shutter is open
4. Try manual Pan/Tilt from dashboard

## Command Line Options

```bash
python dmx_osc_bridge.py [--config CONFIG_FILE]
```

- `--config`: Path to YAML config file (default: `dmx_config.yaml`)

## Adding New Fixtures

### Adding a New Mannequin

1. Add entry to `dmx_config.yaml` under `characters:`
2. Set `dmx_address` to next available (multiples of 8)
3. Assign unique `color`
4. Assign `x32_channel` (Bus 1-6, can share)
5. Restart bridge

### Adding a New House Light

1. Add to appropriate group in `house_lights:`
2. Specify `name`, `dmx_address`, `type`, `channels`
3. For movers, add `home_pan` and `home_tilt`
4. Restart bridge

### Adding Mover Presets

1. Add character entry under `mover_targets:`
2. Specify `{pan, tilt, zoom}` for each mover
3. Add button to dashboard (or use API)
