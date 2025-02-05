<script setup>
import Translator from './components/Translator.vue'
import { ref } from 'vue'

const response = ref('')

async function translate(source) {
	if (source.length !== 0) {
		console.log("Translating: " + source);

		const options = {
			method: "POST",
			headers: { "Content-Type": "text/plain"},
			body: source
		}
		const res = await fetch(`http://127.0.0.1:3456/translate`, options)
			.then(res => res.text())
			.then(text => {
				console.log("Received :" + text);
				response.value = text;
			});
	}
}

</script>

<template>
	<header>
		<img alt="Vue logo" class="logo" src="./assets/logo.svg" width="125" height="125" />

		<h1 class="green" id="main-heading">SUMMIT Translator</h1>
	</header>

	<main>
		<Translator source="German" target="English" :target-text="response" @translate="translate" />
	</main>
	
</template>

<style scoped>
header {
	line-height: 1.5;
	text-align: center;
}

main {
	min-width: 50vw;
}

#main-heading {
	font-size: 3rem;
}

.logo {
	display: block;
	margin: 0 auto 2rem;
}

@media (min-width: 1024px) {
	header {
		display: flex;
		place-items: center;
		padding-right: calc(var(--section-gap) / 2);
	}

	.logo {
		margin: 0 2rem 0 0;
	}

	header .wrapper {
		display: flex;
		place-items: flex-start;
		flex-wrap: wrap;
	}
}
</style>
