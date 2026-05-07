/**
 * Surgical DRS - Modular Player v2.0
 * Handles frame-by-step playback and UI updates.
 */

class DRSPlayer {
    constructor() {
        // UI Elements
        this.frameImg = document.getElementById('main-frame');
        this.playBtn = document.getElementById('play-pause-btn');
        this.playIcon = document.getElementById('play-icon');
        this.pauseIcon = document.getElementById('pause-icon');
        this.playText = document.getElementById('play-text');
        this.stepForwardBtn = document.getElementById('step-forward-btn');
        this.stepBackBtn = document.getElementById('step-back-btn');
        this.resetBtn = document.getElementById('reset-btn');
        this.loader = document.getElementById('loader');
        
        // Telemetry Markers
        this.valFrameIdx = document.getElementById('val-frame-idx');
        this.valPlayback = document.getElementById('val-playback');
        
        // Bowler Config Buttons
        this.btnArmLeft = document.getElementById('btn-arm-left');
        this.btnArmRight = document.getElementById('btn-arm-right');
        this.btnWicketOver = document.getElementById('btn-wicket-over');
        this.btnWicketAround = document.getElementById('btn-wicket-around');
        
        // State
        this.isPlaying = false;
        this.isFetching = false;
        this.playbackInterval = null;
        this.fps = 30;
        this.frameTime = 1000 / this.fps;

        // ROI State
        this.isCalibrating = false;
        this.calMode = 'base'; // 'base' or 'pitching'
        this.roiPoints = [];
        this.roiCanvas = document.getElementById('roi-canvas');
        this.roiCtx = this.roiCanvas.getContext('2d');
        this.calStartBtn = document.getElementById('cal-start-btn');
        this.calPitchBtn = document.getElementById('cal-pitch-btn');
        this.calBowlerBtn = document.getElementById('cal-bowler-btn');
        this.pipelineCells = document.getElementById('pipeline-cells'); // FIX: Automatic pipeline update
        this.calHintText = document.getElementById('cal-hint-text');
        this.calActions = document.getElementById('cal-actions');
        this.calSaveBtn = document.getElementById('cal-save-btn');
        this.calClearBtn = document.getElementById('cal-clear-btn');

        // Debug Pipeline UI - Managed via strip now
        this.debugVisible = false;
        this.pipelineCells = document.getElementById('pipeline-cells');

        this.init();
    }

    init() {
        this.playBtn.addEventListener('click', () => this.togglePlay());
        this.stepForwardBtn.addEventListener('click', () => this.step());
        this.stepBackBtn.addEventListener('click', () => this.stepBack());
        this.resetBtn.addEventListener('click', () => this.reset());

        // ROI Events
        this.calStartBtn.addEventListener('click', () => this.toggleCalibration('base'));
        this.calPitchBtn.addEventListener('click', () => this.toggleCalibration('pitching'));
        this.calBowlerBtn.addEventListener('click', () => this.toggleCalibration('bowler'));
        this.calSaveBtn.addEventListener('click', () => this.saveROI());
        this.calClearBtn.addEventListener('click', () => this.clearROI());
        this.roiCanvas.addEventListener('click', (e) => this.handleROIClick(e));

        // Bowler Config Events
        this.btnArmLeft.addEventListener('click', () => this.setBowlerSide('left'));
        this.btnArmRight.addEventListener('click', () => this.setBowlerSide('right'));
        this.btnWicketOver.addEventListener('click', () => this.setWicketSide('left')); // 'left' in backend is 'over' for righties, but we'll map explicitly
        this.btnWicketAround.addEventListener('click', () => this.setWicketSide('right'));

        // Keyboard Shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.code === 'Space') { e.preventDefault(); this.step(); }
            if (e.code === 'KeyP') { e.preventDefault(); this.togglePlay(); }
            if (e.code === 'ArrowRight') { e.preventDefault(); this.step(); }
            if (e.code === 'ArrowLeft') { e.preventDefault(); this.stepBack(); }
        });

        // Initial Frame + Auto-Play
        this.step().then(() => {
            this.togglePlay();
        });
    }

    log(msg, type = 'info') {
        // Redundant UI logging removed as requested.
        // Redirecting to console for developer audit.
        console.log(`[${type.toUpperCase()}] ${msg}`);
    }

    async reset() {
        this.log("Resetting Video Stream...", "system");
        if (this.isPlaying) this.togglePlay();
        try {
            await fetch('/reset', { method: 'POST' });
            this.valFrameIdx.textContent = '0';
            this.step();
        } catch (e) {
            this.log(`Reset failed: ${e}`, 'system');
        }
    }

    async setBowlerSide(side) {
        this.log(`Setting Bowler Arm: ${side.toUpperCase()}`, 'system');
        try {
            const res = await fetch('/set_bowler_side', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ side })
            });
            const data = await res.json();
            if (data.success) {
                // Update UI state
                this.btnArmLeft.classList.toggle('active', side === 'left');
                this.btnArmRight.classList.toggle('active', side === 'right');
                
                // Refresh frame to show new arm markers
                if (!this.isPlaying) this.refreshCurrentFrame();
            }
        } catch (e) { this.log(`Error setting side: ${e}`, 'system'); }
    }

    async setWicketSide(side) {
        // Map UI labels to backend 'left'/'right'
        // Over wicket (usually left for righties) -> 'left'
        // Around wicket -> 'right'
        this.log(`Setting Wicket Side: ${side === 'left' ? 'OVER' : 'AROUND'}`, 'system');
        try {
            const res = await fetch('/set_wicket_side', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ side })
            });
            const data = await res.json();
            if (data.success) {
                this.btnWicketOver.classList.toggle('active', side === 'left');
                this.btnWicketAround.classList.toggle('active', side === 'right');
                
                // Refresh frame to show new shading/config
                if (!this.isPlaying) this.refreshCurrentFrame();
            }
        } catch (e) { this.log(`Error setting wicket side: ${e}`, 'system'); }
    }

    async refreshCurrentFrame() {
        if (this.isFetching) return;
        this.isFetching = true;
        
        try {
            const res = await fetch('/current_frame');
            const data = await res.json();
            
            if (data.error) {
                this.log(data.error, "system");
                return;
            }

            this.frameImg.src = `data:image/jpeg;base64,${data.image}`;
        } catch (e) {
            this.log(`Error during refresh: ${e}`, 'system');
        } finally {
            this.isFetching = false;
        }
    }

    async stepBack() {
        if (this.isFetching) return;
        this.isFetching = true;
        
        try {
            const res = await fetch('/step_back');
            const data = await res.json();
            
            if (data.error) {
                this.log(data.error, "system");
                this.isFetching = false;
                this.valFrameIdx.innerText = data.frame_idx;
            }
            this.frameImg.src = `data:image/jpeg;base64,${data.image}`;
            
            // AUTOMATIC UPDATES
            this.updatePipeline(data.pipeline);
            
            this.drawROI(data.roi);
        } catch (e) {
            this.log(`Error during step back: ${e}`, 'system');
        } finally {
            this.isFetching = false;
        }
    }

    async step() {
        if (this.isFetching) return;
        this.isFetching = true;
        
        try {
            const res = await fetch('/step');
            const data = await res.json();
            
            if (data.eof) {
                this.log("EOF: End of video reached. Showing final trajectory.", "system");
                if (data.image) {
                    this.frameImg.src = `data:image/jpeg;base64,${data.image}`;
                }
                if (data.pipeline) {
                    this.updatePipeline(data.pipeline); // Persistence for final frame
                }
                if (this.isPlaying) this.togglePlay();
                this.isFetching = false;
                return;
            }

            this.frameImg.src = `data:image/jpeg;base64,${data.image}`;
            this.valFrameIdx.textContent = data.frame_idx;
            
            if (this.loader.style.display !== 'none') {
                this.loader.classList.add('hidden');
            }

            this.refreshDebugPipeline();

        } catch (e) {
            this.log(`Error fetching frame: ${e}`, 'system');
            if (this.isPlaying) this.togglePlay();
        } finally {
            this.isFetching = false;
        }
    }

    togglePlay() {
        this.isPlaying = !this.isPlaying;
        
        if (this.isPlaying) {
            this.playIcon.classList.add('hidden');
            this.pauseIcon.classList.remove('hidden');
            this.playText.textContent = "PAUSE";
            this.valPlayback.textContent = "PLAYING";
            this.valPlayback.classList.add('highlight');
            this.playbackInterval = setInterval(() => this.step(), this.frameTime);
            this.log("Playback Started");
        } else {
            this.playIcon.classList.remove('hidden');
            this.pauseIcon.classList.add('hidden');
            this.playText.textContent = "PLAY";
            this.valPlayback.textContent = "STOPPED";
            this.valPlayback.classList.remove('highlight');
            clearInterval(this.playbackInterval);
            this.log("Playback Paused");
        }
    }

    // --- ROI CALIBRATION LOGIC ---
    toggleCalibration(mode) {
        if (this.isCalibrating && this.calMode !== mode) {
            // Switch mode if already calibrating
            this.calMode = mode;
            this.clearLocalROI();
        } else {
            this.isCalibrating = !this.isCalibrating;
            this.calMode = mode;
        }

        document.body.classList.toggle('calibrating', this.isCalibrating);
        this.calActions.classList.toggle('hidden', !this.isCalibrating);
        
        if (this.isCalibrating) {
            this.log(`Calibration Mode [${mode.toUpperCase()}] ON`, "system");
            const btn = (mode === 'base') ? this.calStartBtn : (mode === 'pitching' ? this.calPitchBtn : this.calBowlerBtn);
            
            // Hide all but selected
            [this.calStartBtn, this.calPitchBtn, this.calBowlerBtn].forEach(b => {
                b.classList.add('hidden');
            });
            btn.classList.remove('hidden');
            btn.textContent = "CANCEL";
            
            const hints = {
                'base': "Click 4 points for the PITCH BASE",
                'pitching': "Click 4 points for the BLUE IMPACT ZONE",
                'bowler': "Click 2 points for the VERTICAL RELEASE LANE"
            };
            this.calHintText.textContent = hints[mode];
            this.calActions.classList.remove('hidden'); // Ensure visible
            
            this.syncCanvasSize();
        } else {
            this.log("Calibration Mode OFF");
            this.calStartBtn.innerHTML = "<span class='icon'>📐</span> MAIN ROI (YELLOW)";
            this.calPitchBtn.innerHTML = "<span class='icon'>🟦</span> PITCHING ZONE (BLUE)";
            this.calBowlerBtn.innerHTML = "<span class='icon'>🟩</span> BOWLER ROI (GREEN)";
            
            [this.calStartBtn, this.calPitchBtn, this.calBowlerBtn].forEach(b => b.classList.remove('hidden'));
            this.calActions.classList.add('hidden'); // Explicit hide
            this.clearLocalROI();
        }
    }

    syncCanvasSize() {
        const rect = this.frameImg.getBoundingClientRect();
        this.roiCanvas.width = rect.width;
        this.roiCanvas.height = rect.height;
        this.roiCanvas.style.left = `${this.frameImg.offsetLeft}px`;
        this.roiCanvas.style.top = `${this.frameImg.offsetTop}px`;
    }

    handleROIClick(e) {
        if (!this.isCalibrating) return;

        const rect = this.roiCanvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        // Dynamic Mapping to the actual resolution of the frame received from the server
        const sourceX = Math.round((x / rect.width) * this.frameImg.naturalWidth);
        const sourceY = Math.round((y / rect.height) * this.frameImg.naturalHeight);

        this.roiPoints.push([sourceX, sourceY]);
        this.drawROIPreview();
    }

    drawROIPreview() {
        const rect = this.roiCanvas.getBoundingClientRect();
        const ctx = this.roiCtx;
        ctx.clearRect(0, 0, this.roiCanvas.width, this.roiCanvas.height);
        
        if (this.roiPoints.length === 0) return;

        const natW = this.frameImg.naturalWidth || 640;
        const natH = this.frameImg.naturalHeight || 640;

        const colors = { 'base': "#fbbf24", 'pitching': "#3b82f6", 'bowler': "#22c55e" };
        const color = colors[this.calMode] || "#fbbf24";
        
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        ctx.lineWidth = 2;

        const displayPoints = this.roiPoints.map(p => ({
            x: (p[0] / natW) * rect.width,
            y: (p[1] / natH) * rect.height
        }));

        // Draw Base
        ctx.beginPath();
        if (displayPoints.length > 0) ctx.moveTo(displayPoints[0].x, displayPoints[0].y);
        for(let i=1; i<displayPoints.length; i++) {
            ctx.lineTo(displayPoints[i].x, displayPoints[i].y);
        }
        if (this.roiPoints.length > 2) ctx.closePath();
        ctx.stroke();

        // Draw Vertical lines and Top Lid (3D look - Extended to Top)
        const topY = 0; // Top of browser viewport
        if (displayPoints.length > 1) {
            ctx.beginPath();
            ctx.moveTo(displayPoints[0].x, topY);
            for(let i=1; i<displayPoints.length; i++) {
                ctx.lineTo(displayPoints[i].x, topY);
            }
            if (this.roiPoints.length > 2) ctx.closePath();
            ctx.stroke();
        }

        if (this.calMode === 'base') {
            displayPoints.forEach(p => {
                // Draw Vertical Pillar to Top
                ctx.beginPath();
                ctx.moveTo(p.x, p.y);
                ctx.lineTo(p.x, topY);
                ctx.stroke();
                
                // Draw Vertex Joints (Base Only)
                ctx.beginPath();
                ctx.arc(p.x, p.y, 4, 0, Math.PI*2);
                ctx.fill();
            });
        } else if (this.calMode === 'bowler') {
            // 2-POINT VERTICAL RIBBON
            displayPoints.forEach(p => {
                ctx.beginPath();
                ctx.moveTo(p.x, p.y);
                ctx.lineTo(p.x, 0); // Straight to top
                ctx.stroke();
                ctx.beginPath();
                ctx.arc(p.x, p.y, 5, 0, Math.PI*2);
                ctx.fill();
            });
        } else {
            // Just draw corners for pitching zone (4 pts)
            displayPoints.forEach(p => {
                ctx.beginPath();
                ctx.arc(p.x, p.y, 5, 0, Math.PI*2);
                ctx.fill();
            });
        }
    }

    clearLocalROI() {
        this.roiPoints = [];
        this.roiCtx.clearRect(0, 0, this.roiCanvas.width, this.roiCanvas.height);
    }

    async clearROI() {
        await fetch('/clear_roi', { method: 'POST' });
        this.clearLocalROI();
        this.log("ROI Cleared from Server", "system");
        this.step();
    }

    async saveROI() {
        const minPoints = (this.calMode === 'bowler') ? 2 : 4;
        if (this.roiPoints.length < minPoints) {
            return alert(`Need ${minPoints} points for this zone!`);
        }
        
        const res = await fetch('/set_roi', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                points: this.roiPoints,
                type: this.calMode 
            })
        });
        
        if (res.ok) {
            this.log(`ROI [${this.calMode.toUpperCase()}] Saved!`, "success");
            this.toggleCalibration(this.calMode);
            this.fetchFrame(); // Instant Refresh
        }
    }

    // --- DEBUG PIPELINE VIEW ---
    toggleDebug() {
        console.log("ToggleDebug is legacy - Pipeline is now permanent.");
    }

    updatePipeline(stages) {
        if (!stages) return;
        
        // Update each cell in the strip synchronously
        for (const [key, b64] of Object.entries(stages)) {
            const cell = document.getElementById(`cell-${key}`);
            if (cell) {
                const img = cell.querySelector('.cell-img');
                if (img) img.src = `data:image/jpeg;base64,${b64}`;
            }
        }
    }
}

// Initialize on Load
window.addEventListener('DOMContentLoaded', () => {
    window.player = new DRSPlayer();
});
