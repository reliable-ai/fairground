<script setup>
import { transformerNotationHighlight } from "@shikijs/transformers";
import { codeToHtml } from "shiki";
import { onMounted, ref, watch } from "vue";

const props = defineProps({
  code: {
    type: String,
    required: true,
  },
  language: {
    type: String,
    default: "python",
  },
});

const emit = defineEmits(["copy"]);

const copied = ref(false);
const highlightedCode = ref("");
const isHighlighting = ref(true);

const copyCode = () => {
  navigator.clipboard.writeText(props.code);
  copied.value = true;
  emit("copy");

  setTimeout(() => {
    copied.value = false;
  }, 2000);
};

const highlightCode = async (codeToHighlight) => {
  isHighlighting.value = true;
  try {
    highlightedCode.value = await codeToHtml(codeToHighlight, {
      lang: props.language,
      themes: {
        light: "github-light",
        dark: "github-dark",
      },
      transformers: [transformerNotationHighlight()],
    });
  } catch (error) {
    console.error("Error highlighting code:", error);
    // Fallback to plain text if highlighting fails
    highlightedCode.value = `<pre class="language-${props.language}"><code>${codeToHighlight}</code></pre>`;
  }
  isHighlighting.value = false;
};

onMounted(() => {
  highlightCode(props.code);
});

// Watch for changes to the code and update highlighting
watch(
  () => props.code,
  (newCode) => {
    highlightCode(newCode);
  },
);
</script>

<template>
  <div class="code-block-wrapper" :class="`language-${language}`">
    <div v-if="isHighlighting" class="loading-code">
      Loading code preview...
    </div>
    <div v-else v-html="highlightedCode" class="shiki-code-block"></div>
    <button class="copy" :class="{ copied }" @click="copyCode"></button>
  </div>
</template>

<style scoped>
.code-block-wrapper {
  position: relative;
  margin-top: 16px;
  border-radius: 8px;
  overflow: hidden;
}

.loading-code {
  background-color: var(--vp-code-block-bg, #1e1e1e);
  color: var(--vp-c-text-2);
  padding: 20px;
  border-radius: 8px;
  text-align: center;
}

:deep(.shiki) {
  background-color: transparent !important;
}

.copy {
  position: absolute;
  top: 8px;
  right: 8px;
  width: 40px;
  height: 40px;
  padding: 0;
  border: 0;
  cursor: pointer;
  background-color: transparent;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' height='20' width='20' stroke='rgba(128,128,128,1)' stroke-width='2' viewBox='0 0 24 24'%3E%3Cpath stroke-linecap='round' stroke-linejoin='round' d='M9 5H7a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2h-2M9 5a2 2 0 0 0 2 2h2a2 2 0 0 0 2-2M9 5a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2'/%3E%3C/svg%3E");
  background-position: center;
  background-repeat: no-repeat;
  transition: all 0.3s;
  opacity: 0.5;
}

.copy:hover {
  opacity: 1;
}

.copy.copied {
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' height='20' width='20' stroke='rgba(40,167,69,1)' stroke-width='2' viewBox='0 0 24 24'%3E%3Cpath stroke-linecap='round' stroke-linejoin='round' d='M5 13l4 4L19 7'/%3E%3C/svg%3E");
}
</style>