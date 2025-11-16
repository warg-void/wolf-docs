<script setup lang="ts">
import ParentLayout from '@vuepress/theme-default/layouts/Layout.vue'
</script>

<template>
  <ParentLayout>
    <template #page-bottom>
      <footer class="wolf-footer">
        <p>Made with ❤️ for C++</p>
      </footer>
    </template>
  </ParentLayout>
</template>

<style>
.wolf-footer {
  margin-top: 2rem;
  padding: 1rem 0;
  text-align: center;
  font-size: 0.9rem;
  opacity: 0.7;
  border-top: 1px solid var(--vp-c-border, #444);
}
</style>
