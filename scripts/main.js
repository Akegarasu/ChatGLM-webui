// Gradio ui update hook from stable-diffusion-webui-chinese


var _app;
function gradioApp(){
	return _app||(_app=document.getElementsByTagName('gradio-app')[0].getRootNode());
}

function get_uiCurrentTab() {
    return gradioApp().querySelector('.tabs button:not(.border-transparent)')
}

function get_uiCurrentTabContent() {
    return gradioApp().querySelector('.tabitem[id^=tab_]:not([style*="display: none"])')
}

let onupdates = [], ontabchanges = [], onloads = [];
let uiCurrentTab = null;

function onUiUpdate(fn){
	onupdates.push(fn);
}
function onUiTabChange(fn){
	ontabchanges.push(fn);
}
function onLoad(fn) {
	onloads?onloads.push(fn):fn();
}

function runCallback(x, m) {
	for(let i=0;i<x.length;i++) {
		try {
			x[i](m)
		} catch (e) {
			(console.error || console.log).call(console, e.message, e);
		}
	}
}

document.addEventListener("DOMContentLoaded", function() {
	let debounce = setInterval(function () {
		if (!(uiCurrentTab = get_uiCurrentTab())) return;
		clearInterval(debounce);

		console.clear && console.clear();

		onloads.forEach((e) => e());
		onloads = null;

		moRunning.observe(gradioApp(), {childList: true, subtree: true});
	}, 10);

	let moRunning = new MutationObserver(function(m) {
		clearTimeout(debounce);
		debounce = setTimeout(function() {
			runCallback(onupdates,m);
			const newTab = get_uiCurrentTab();
			if (newTab !== uiCurrentTab) {
				uiCurrentTab = newTab;
				runCallback(ontabchanges,m);
			}
		}, 20);
	});
});

let generate_button;
onLoad(() => {
	// c_generate
	let base = get_uiCurrentTabContent();
	generate_button = base.querySelector('button[id$=_generate]');
	let textarea = base.querySelector("#chat-input textarea")
	generate_button.addEventListener('click', () => {
		textarea.value = "";
		// update svelte internal state
		textarea.dispatchEvent(new InputEvent("input"));
	});
});

/**
 * Add a ctrl+enter as a shortcut to start a generation
 */
document.addEventListener('keydown', function(e) {
	if (e.key) {
		if (!(e.key === "Enter" && (e.metaKey || e.ctrlKey))) return
	} else if (e.keyCode) {
		if (!(e.keyCode === 13 && (e.metaKey || e.ctrlKey))) return;
	} else return;

	generate_button.click();
	e.preventDefault();
});
