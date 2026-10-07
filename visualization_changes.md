# Visualization and UI Changes

Based on the recent git commits, several updates were made to the performance sidebar visualization to better support and display distributed grid execution. Here is a comprehensive list of all the changes made:

## 1. Device Role Identification (`room.js`)
*   **Worker Role Property**: Added a new `workerRole` property to devices within the cluster. Based on the active AI configuration and layer assignments, devices are now dynamically classified as **"Host"**, **"Worker"**, or **"Idle"**.

## 2. Main Sidebar & Hero Updates (`room/perf-sidebar.js`)
*   **Dynamic Execution Subtitle**: The hero section now intelligently checks the number of active nodes (devices with an active worker role). 
    *   If multiple nodes are participating, it displays **"Distributed Execution"**.
    *   If only one node is active, it defaults to **"Solo Device Execution"**.
*   **Grid Participation Label**: Changed the section header label from `"GRID TOKENS PROCESSED"` to `"GRID PARTICIPATION"`.
*   **Dynamic Node Badge**: The node count badge (`perf-grid-share-badge`) now dynamically updates to show the correct number of active participating nodes (e.g., "1 NODE" vs "3 NODES").

## 3. Device Card Enhancements (`room/perf-sidebar.js`)
*   **Role Badges**: The badge on individual device cards that previously showed the current "stage" now displays the device's specific distributed role:
    *   Displays **"Distributed Worker"** if it is a Worker.
    *   Displays **"Host"** if it is the Host.
    *   Falls back to displaying the current stage or **"Idle"** if neither.
*   **Contextual Metrics Display**: The metrics displayed on the right side of the device cards now change based on the device's role:
    *   **Workers**: Display the specific **model layers** they have been assigned (e.g., `layers 0-15`).
    *   **Hosts**: Continue to display the live **Tokens per Second (tok/s)** and **Total Tokens (tok)** processed.

## 4. Telemetry and Charting Fixes (`room/perf-sidebar.js` & `patch2.py`)
*   **Host Telemetry Attribution**: Updated the telemetry logic that records streaming data points for the TPS charts. Previously, these metrics were hardcoded to always attribute to the local `"self"` device. It has been updated to dynamically find the `"Host"` device (falling back to `"self"` if needed) and attribute the tokens and TPS to them, ensuring the charts render accurately during distributed generation.
*   *Note: A Python patch script (`patch2.py`) was also committed to automate the text replacement for this specific telemetry fix.*
