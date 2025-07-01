<template>
  <div id="app">
    <Home v-if="showHome" @connect="handleConnect" />
    <EvaluateForm v-else @back="handleBack" />
  </div>
</template>

<script>
import Home from './components/Home.vue';
import EvaluateForm from './components/EvaluateForm.vue';

export default {
  components: { Home, EvaluateForm },
  data() {
    return {
      showHome: true,
    };
  },
  methods: {
    handleConnect() {
      this.showHome = false;
    },
    handleBack() {
      this.showHome = true;
      // 如果想重新加载Home组件，可以通过改变key，或者重置相关数据
    },
  },
};
</script>
