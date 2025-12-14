const canvas = document.getElementById('isingCanvas');
const ctx = canvas.getContext('2d');
const N = 128; // lattice size
canvas.width = N;
canvas.height = N;
ctx.imageSmoothingEnabled = false;

const spins = new Int8Array(N * N);
const imageData = ctx.createImageData(N, N);
const pixels = imageData.data;

let temp = 2.27;
let sweepsPerFrame = 8;
let running = false;
let magnetization = 0;
let energy = 0;
let accepted = 0;
let attempts = 0;
let p4 = Math.exp(-4 / temp);
let p8 = Math.exp(-8 / temp);

const magCtx = document.getElementById('magChart').getContext('2d');
Chart.defaults.color = '#cbd5e1';
Chart.defaults.font.family = "'Space Grotesk','DM Sans',sans-serif";
const magChart = new Chart(magCtx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [
            {
                label: 'Magnetization / spin',
                data: [],
                borderColor: '#22d3ee',
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.12,
                fill: false,
            },
        ],
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        scales: {
            x: { display: false },
            y: {
                min: -1,
                max: 1,
                grid: { color: '#1f2b3f' },
            },
        },
        plugins: { legend: { display: false } },
    },
});

function idx(x, y) {
    return x * N + y;
}

function computeEnergy() {
    let E = 0;
    for (let x = 0; x < N; x++) {
        for (let y = 0; y < N; y++) {
            const s = spins[idx(x, y)];
            const right = spins[idx(x, (y + 1) % N)];
            const down = spins[idx((x + 1) % N, y)];
            E -= s * (right + down);
        }
    }
    return E;
}

function computeMagnetization() {
    let m = 0;
    for (let i = 0; i < spins.length; i++) {
        m += spins[i];
    }
    return m;
}

function updateBoltzmann() {
    p4 = Math.exp(-4 / temp);
    p8 = Math.exp(-8 / temp);
}

function initializeLattice() {
    for (let i = 0; i < spins.length; i++) {
        spins[i] = Math.random() > 0.5 ? 1 : -1;
    }
    energy = computeEnergy();
    magnetization = computeMagnetization();
    accepted = 0;
    attempts = 0;
    updateBoltzmann();
    renderLattice();
    updateStats();
    resetChart();
}

function metropolisSweep() {
    const total = N * N * sweepsPerFrame;
    let acceptedNow = 0;
    for (let t = 0; t < total; t++) {
        const x = (Math.random() * N) | 0;
        const y = (Math.random() * N) | 0;
        const id = idx(x, y);
        const s = spins[id];

        const up = idx((x - 1 + N) % N, y);
        const down = idx((x + 1) % N, y);
        const left = idx(x, (y - 1 + N) % N);
        const right = idx(x, (y + 1) % N);

        const nb = spins[up] + spins[down] + spins[left] + spins[right];
        const dE = 2 * s * nb;

        let accept = false;
        if (dE <= 0) {
            accept = true;
        } else if (dE === 4) {
            accept = Math.random() < p4;
        } else if (dE === 8) {
            accept = Math.random() < p8;
        }

        if (accept) {
            spins[id] = -s;
            energy += dE;
            magnetization -= 2 * s;
            acceptedNow++;
        }
    }
    accepted += acceptedNow;
    attempts += total;
}

function renderLattice() {
    const upColor = [224, 252, 255];
    const downColor = [13, 23, 42];
    const len = spins.length;

    for (let i = 0; i < len; i++) {
        const color = spins[i] === 1 ? upColor : downColor;
        const p = i * 4;
        pixels[p] = color[0];
        pixels[p + 1] = color[1];
        pixels[p + 2] = color[2];
        pixels[p + 3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);
}

function updateStats() {
    const norm = N * N;
    const ratio = attempts ? accepted / attempts : 0;
    document.getElementById('energyVal').textContent = (energy / norm).toFixed(3);
    document.getElementById('magVal').textContent = (Math.abs(magnetization) / norm).toFixed(3);
    document.getElementById('acceptanceVal').textContent = (ratio * 100).toFixed(1) + '%';
    document.getElementById('tempDisplay').textContent = temp.toFixed(2);
    document.getElementById('tempLabel').textContent = temp.toFixed(2);
    document.getElementById('speedLabel').textContent = sweepsPerFrame.toString();
}

function resetChart() {
    magChart.data.labels = [];
    magChart.data.datasets[0].data = [];
    magChart.update();
}

function updateChart() {
    const magPerSpin = magnetization / (N * N);
    magChart.data.labels.push('');
    magChart.data.datasets[0].data.push(magPerSpin);
    if (magChart.data.labels.length > 300) {
        magChart.data.labels.shift();
        magChart.data.datasets[0].data.shift();
    }
    magChart.update('none');
}

function loop() {
    if (!running) return;
    metropolisSweep();
    renderLattice();
    updateStats();
    updateChart();
    requestAnimationFrame(loop);
}

// Event bindings
document.getElementById('tempSlider').addEventListener('input', (e) => {
    temp = parseFloat(e.target.value);
    updateBoltzmann();
    updateStats();
});

document.getElementById('speedSlider').addEventListener('input', (e) => {
    sweepsPerFrame = parseInt(e.target.value, 10);
    updateStats();
});

document.getElementById('btnStart').addEventListener('click', () => {
    if (!running) {
        running = true;
        loop();
    }
});

document.getElementById('btnStop').addEventListener('click', () => {
    running = false;
});

document.getElementById('btnReset').addEventListener('click', () => {
    running = false;
    initializeLattice();
});

// Bootstrap
document.getElementById('latticeMeta').textContent = `Lattice: ${N} × ${N} spins`;
initializeLattice();
