<template>
    <div class="page-wrapper" style="display:flex; flex-direction: column; gap: 1em; padding: 1em;">
        <!-- 页面标题栏 -->
        <div class="page-header">
            <h2 class="page-title">🎙 英语口语练习平台</h2>
        </div>

        <!-- 个人练习情况标题 -->
        <h2>个人练习情况</h2>

        <div style="display: flex; gap: 2em; align-items: flex-start;">
            <!-- 左侧仪表盘 -->
            <div style="flex: 1; display: flex; justify-content: space-around;">
                <div ref="accuracyGauge" style="width: 200px; height: 200px;"></div>
                <div ref="fluencyGauge" style="width: 200px; height: 200px;"></div>
                <div ref="standardGauge" style="width: 200px; height: 200px;"></div>
            </div>

            <!-- 右侧：折线图 + 按钮 -->
            <div style="flex: 2; display: flex; flex-direction: column; align-items: center; min-width: 600px;">
                <!-- 折线图容器 保持宽度和高度 -->
                <div ref="totalScoreLine"
                     style="width: 100%; height: 280px; border: 1px solid #ccc; margin-bottom: 1em;"></div>

                <!-- 切换按钮 -->
                <button class="action-btn" @click="toggleChart" style="width: 200px;">
                    切换为{{ showAverage ? '总分' : '平均分' }}折线图
                </button>
            </div>
        </div>


        <!-- 历史练习记录 -->
        <div class="history-list" style="margin-top: 1em;">
            <h3>近五次历史练习评分：</h3>
            <ul>
                <li v-for="rec in records.slice(0,5)" :key="rec.filename">
                    {{ formatTimestamp(rec.mtime) }} —
                    总分: {{ rec.totalScore }}，
                    准确度: {{ parseFloat(rec.accuracyScore).toFixed(1) }}，
                    流利度: {{ parseFloat(rec.fluencyScore).toFixed(1) }}，
                    标准度: {{ parseFloat(rec.standardScore).toFixed(1) }}
                </li>
            </ul>
        </div>



        <!-- 开始练习按钮 -->
        <button class="action-btn submit-btn" @click="$emit('connect')">开始练习</button>
    </div>
</template>

<script>
    import * as echarts from 'echarts';

    function gaugeOption(title, value) {
        return {
            title: {
                text: title,
                left: 'center',
                top: '5%'
            },
            series: [
                {
                    type: 'gauge',
                    max: 5,
                    center: ['50%', '60%'],
                    progress: {
                        show: value,
                        width: 18
                    },
                    axisLine: {
                        lineStyle: {
                            width: 18,
                            color: [
                                [0.4, '#FF4C4C'], // 红：低分
                                [0.8, '#FFA500'], // 橙：中分
                                [1.0, '#4CAF50']  // 绿：高分
                            ]
                        }
                    },
                    axisTick: { show: false },
                    splitLine: { length: 10 },
                    axisLabel: { distance: 15 },
                    pointer: { width: 5 },
                    detail: {
                        valueAnimation: true,
                        formatter: (val) => val.toFixed(3),
                        fontSize: 20,
                        offsetCenter: [0, '75%']
                    },
                    data: [{ value: value }]
                }
            ]
        };
    }


    export default {
        name: "Home",
        data() {
            return {
                loading: false,
                error: null,
                records: [],
                currentAccuracy: 0,
                currentFluency: 0,
                currentStandard: 0,
                accuracyChart: null,
                fluencyChart: null,
                standardChart: null,
                lineChart: null,
                showAverage: false
            };
        },
        mounted() {
            this.loadRecordsFromBackend();
            window.addEventListener("resize", this.handleResize);
        },
        beforeUnmount() {
            window.removeEventListener("resize", this.handleResize);
            this.accuracyChart?.dispose();
            this.fluencyChart?.dispose();
            this.standardChart?.dispose();
            this.lineChart?.dispose();
        },
        methods: {
            handleResize() {
                this.accuracyChart?.resize();
                this.fluencyChart?.resize();
                this.standardChart?.resize();
                this.lineChart?.resize();
            },

            toggleChart() {
                this.showAverage = !this.showAverage;
                this.renderLineChart();
            },

            formatTimestamp(ts) {
                const dt = new Date(ts * 1000);
                return dt.toLocaleString("zh-CN");
            },

            async loadRecordsFromBackend() {
                this.loading = true;
                this.error = null;

                try {
                    const res = await fetch("http://localhost:5000/api/history");
                    if (!res.ok) throw new Error("获取历史记录失败");
                    const data = await res.json();
                    this.records = data;

                    if (data.length > 0) {
                        const total = data.reduce(
                            (sum, r) => {
                                sum.acc += parseFloat(r.accuracyScore) || 0;
                                sum.flu += parseFloat(r.fluencyScore) || 0;
                                sum.std += parseFloat(r.standardScore) || 0;
                                return sum;
                            },
                            { acc: 0, flu: 0, std: 0 }
                        );

                        const n = data.length;
                        this.currentAccuracy = (total.acc / n).toFixed(3);
                        this.currentFluency = (total.flu / n).toFixed(3);
                        this.currentStandard = (total.std / n).toFixed(3);
                    }


                    this.$nextTick(() => {
                        this.renderGauges();
                        this.renderLineChart();
                    });
                } catch (err) {
                    this.error = "加载失败：" + err.message;
                } finally {
                    this.loading = false;
                }
            },

            renderGauges() {
                if (!this.accuracyChart) {
                    this.accuracyChart = echarts.init(this.$refs.accuracyGauge);
                }
                this.accuracyChart.setOption(gaugeOption("准确度", this.currentAccuracy));

                if (!this.fluencyChart) {
                    this.fluencyChart = echarts.init(this.$refs.fluencyGauge);
                }
                this.fluencyChart.setOption(gaugeOption("流利度", this.currentFluency));

                if (!this.standardChart) {
                    this.standardChart = echarts.init(this.$refs.standardGauge);
                }
                this.standardChart.setOption(gaugeOption("标准度", this.currentStandard));
            },

            renderLineChart() {
                if (!this.lineChart) {
                    this.lineChart = echarts.init(this.$refs.totalScoreLine);
                }

                const sortedRecords = [...this.records].sort((a, b) => a.mtime - b.mtime);
                const times = sortedRecords.map(r => this.formatTimestamp(r.mtime));

                let scores = [];
                let title = "";

                if (this.showAverage) {
                    title = "平均分历史趋势（逐次累计）";

                    let total = 0;
                    let count = 0;

                    for (const r of sortedRecords) {
                        const val = parseFloat(r.totalScore);
                        if (!isNaN(val)) {
                            total += val;
                            count += 1;
                            scores.push((total / count).toFixed(2));
                        } else {
                            // 若该条无效，仍需保持索引一致，可插入上一个平均值或 0
                            scores.push(count > 0 ? (total / count).toFixed(2) : 0);
                        }
                    }
                } else {
                    title = "总分历史趋势";
                    scores = sortedRecords.map(r => {
                        const val = parseFloat(r.totalScore);
                        return isNaN(val) ? 0 : val;
                    });
                }

                const option = {
                    title: { text: title, left: "center" },
                    tooltip: { trigger: "axis" },
                    xAxis: {
                        type: "category",
                        data: times,
                        axisLabel: { rotate: 45, fontSize: 10 }
                    },
                    yAxis: {
                        type: "value",
                        min: 0,
                        max: 5,
                        axisLabel: { formatter: "{value}" }
                    },
                    series: [
                        {
                            data: scores,
                            type: "line",
                            smooth: true,
                            areaStyle: {}
                        }
                    ]
                };

                this.lineChart.setOption(option);
            }


        }
    };
</script>





<style scoped>
    .page-wrapper {
        min-height: 100vh;
        width: 100%;
        display: flex;
        flex-direction: column;
        align-items: center;
        padding-top: 60px;
        box-sizing: border-box;
        position: relative;
        z-index: 0;
    }

        .page-wrapper::before {
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: url("/background.jpg") center/cover no-repeat;
            background-attachment: fixed;
            z-index: -1;
        }

    .page-header {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 60px;
        background: rgba(255, 255, 255, 0.5);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        z-index: 10;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .page-title {
        color: rgb(0, 0, 0);
        font-size: clamp(22px, 2.5vw, 32px);
        margin: 0;
        white-space: nowrap;
    }

    .home-card {
        margin-top: 50px;
        width: 90vw;
        max-width: 600px;
        background: rgba(255, 255, 255, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        padding: 40px 30px;
        border-radius: 20px;
        text-align: center;
        color: #333;
    }

        .home-card h1 {
            font-size: clamp(24px, 2.5vw, 32px);
            margin-bottom: 30px;
        }

    .action-btn {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 0.8em 1.6em;
        font-size: clamp(14px, 1vw, 16px);
        font-weight: bold;
        background: linear-gradient(135deg, #6DC1F7 0%, #BCF1F2 100%);
        border: none;
        border-radius: 10px;
        color: rgb(0, 0, 0);
        cursor: pointer;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15);
        transition: transform 0.2s ease, box-shadow 0.3s ease;
    }

        .action-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 14px rgba(0, 0, 0, 0.2);
        }

        .action-btn:active {
            transform: scale(0.97);
        }
</style>
