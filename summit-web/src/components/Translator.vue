<script setup>
import { ref } from 'vue'

defineProps({
	source: {
		type: String,
		required: false,
	},
	target: {
		type: String,
		required: false,
	},
	targetText: {
		type: String,
		required: false,
	}
})
const sourceText = ref('')

const emit = defineEmits(['translate'])
</script>

<template>
	<div class="container">
		<div class="field-container">
			<h1 class="green">{{ source }}</h1>
			<input v-model="sourceText" placeholder="Schreibe einen deutschen Satz..."></input>
		</div>
		<div class="field-container">
			<h1 class="green">{{ target }}</h1>
			<input :value="targetText" placeholder="...to get its English translation here" disabled></input>
		</div>
		<button @click="emit('translate', sourceText)">Translate</button>
	</div>
</template>

<style scoped>

.container {
	padding: 1rem;
	background-color: darkslategray;
	border-radius: 0.5rem;
}

.field-container, button {
	margin: 1rem 0;
	text-align: center;
}

input, button {
	width: 100%;
	height: 4rem;
	font-size: 1.5rem;
	font-weight: 400;
	background-color: darkslategray;
	color: whitesmoke;
	border: 1px solid whitesmoke;
	border-radius: 0.5rem;
	padding: 0 1rem;
	text-align: center;
}

button {
	background-color: slategray;
}

input:hover:enabled, input:active:enabled {
	background-color: rgb(37, 61, 61);
}

button:hover {
	background-color: lightslategray;
}

button:active {
	background-color: lightslategray;
	border: 1px solid gray;
	color: gray;
}

@media (min-width: 1024px) {

	
}
</style>
