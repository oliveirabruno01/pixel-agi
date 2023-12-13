function gradioApp() {
    const elems = document.getElementsByTagName('gradio-app');
    const elem = elems.length == 0 ? document : elems[0];

    if (elem !== document) {
        elem.getElementById = function(id) {
            return document.getElementById(id);
        };
    }
    return elem.shadowRoot ? elem.shadowRoot : elem;
}


var uiUpdateCallbacks = [];
var uiAfterUpdateCallbacks = [];
var uiLoadedCallbacks = [];
var uiTabChangeCallbacks = [];
var optionsChangedCallbacks = [];
var uiAfterUpdateTimeout = null;
var uiCurrentTab = null;

/**
 * Register callback to be called at each UI update.
 * The callback receives an array of MutationRecords as an argument.
 */
function onUiUpdate(callback) {
    uiUpdateCallbacks.push(callback);
}

/**
 * Register callback to be called soon after UI updates.
 * The callback receives no arguments.
 *
 * This is preferred over `onUiUpdate` if you don't need
 * access to the MutationRecords, as your function will
 * not be called quite as often.
 */
function onAfterUiUpdate(callback) {
    uiAfterUpdateCallbacks.push(callback);
}

/**
 * Register callback to be called when the UI is loaded.
 * The callback receives no arguments.
 */
function onUiLoaded(callback) {
    uiLoadedCallbacks.push(callback);
}

/**
 * Register callback to be called when the UI tab is changed.
 * The callback receives no arguments.
 */
function onUiTabChange(callback) {
    uiTabChangeCallbacks.push(callback);
}

/**
 * Register callback to be called when the options are changed.
 * The callback receives no arguments.
 * @param callback
 */
function onOptionsChanged(callback) {
    optionsChangedCallbacks.push(callback);
}

function executeCallbacks(queue, arg) {
    for (const callback of queue) {
        try {
            callback(arg);
        } catch (e) {
            console.error("error running callback", callback, ":", e);
        }
    }
}

/**
 * Schedule the execution of the callbacks registered with onAfterUiUpdate.
 * The callbacks are executed after a short while, unless another call to this function
 * is made before that time. IOW, the callbacks are executed only once, even
 * when there are multiple mutations observed.
 */
function scheduleAfterUiUpdateCallbacks() {
    clearTimeout(uiAfterUpdateTimeout);
    uiAfterUpdateTimeout = setTimeout(function() {
        executeCallbacks(uiAfterUpdateCallbacks);
    }, 200);
}

var executedOnLoaded = false;

document.addEventListener("DOMContentLoaded", function() {
    var mutationObserver = new MutationObserver(function(m) {
        if (!executedOnLoaded && gradioApp().querySelector('#prompt_input')) {
            executedOnLoaded = true;
            executeCallbacks(uiLoadedCallbacks);
        }

        executeCallbacks(uiUpdateCallbacks, m);
        scheduleAfterUiUpdateCallbacks();
//        const newTab = get_uiCurrentTab();
//        if (newTab && (newTab !== uiCurrentTab)) {
//            uiCurrentTab = newTab;
//            executeCallbacks(uiTabChangeCallbacks);
//        }
    });
    mutationObserver.observe(gradioApp(), {childList: true, subtree: true});
});

/**
 * Add a ctrl+enter as a shortcut to start a generation
 */
document.addEventListener('keydown', function(e) {
    var handled = false;
    if (e.key !== undefined) {
        if ((e.key == "Enter" && (e.metaKey || e.ctrlKey || e.altKey))) handled = true;
    } else if (e.keyCode !== undefined) {
        if ((e.keyCode == 13 && (e.metaKey || e.ctrlKey || e.altKey))) handled = true;
    }
    if (handled) {
        var button = get_uiCurrentTabContent().querySelector('button[id$=_generate]');
        if (button) {
            button.click();
        }
        e.preventDefault();
    }
});


// Style for new elements. Gets appended to the Gradio root.
const autocompleteCSS_dark = `
    #autocompleteResults {
        position: fixed;
        z-index: 999;
        margin: 5px 0 0 0;
        background-color: #0b0f19 !important;
        border: 1px solid #4b5563 !important;
        border-radius: 12px !important;
        overflow: hidden;
    }
    #autocompleteResultsList > li:nth-child(odd) {
        background-color: #111827;
    }
    #autocompleteResultsList > li {
        list-style-type: none;
        padding: 10px;
        cursor: pointer;
    }
    #autocompleteResultsList > li:hover {
        background-color: #1f2937;
    }
    #autocompleteResultsList > li.selected {
        background-color: #374151;
    }
`;

var acConfig = null;

function parseCSV(str) {
    return str.split('\n').map(line =>
        line.split(',').map((item, index) =>
            index === 0 ? `&lt;img:${item}&gt;` : item
        )
    );
}


// Debounce function to prevent spamming the autocomplete function
var dbTimeOut;
const debounce = (func, wait = 300) => {
    return function(...args) {
        if (dbTimeOut) {
            clearTimeout(dbTimeOut);
        }

        dbTimeOut = setTimeout(() => {
            func.apply(this, args);
        }, wait);
    }
}

// Difference function to fix duplicates not being seen as changes in normal filter
function difference(a, b) {
    return [...b.reduce( (acc, v) => acc.set(v, (acc.get(v) || 0) - 1),
            a.reduce( (acc, v) => acc.set(v, (acc.get(v) || 0) + 1), new Map() )
    )].reduce( (acc, [v, count]) => acc.concat(Array(Math.abs(count)).fill(v)), [] );
}

// Create the result list div and necessary styling
function createResultsDiv() {
    let resultsDiv = document.createElement("div");
    let resultsList = document.createElement('ul');

    resultsDiv.setAttribute('id', 'autocompleteResults');
    resultsList.setAttribute('id', 'autocompleteResultsList');
    resultsDiv.appendChild(resultsList);

    return resultsDiv;
}

// The selected tag index. Needs to be up here so hide can access it.
var selectedTag = null;

// Show or hide the results div
var isVisible = false;
function showResults() {
    let resultsDiv = gradioApp().querySelector('#autocompleteResults');
    resultsDiv.style.display = "block";
    isVisible = true;
}
function hideResults() {
    let resultsDiv = gradioApp().querySelector('#autocompleteResults');
    resultsDiv.style.display = "none";
    isVisible = false;
    selectedTag = null;
}

function htmlDecode(input){
  var doc = new DOMParser().parseFromString(input, "text/html");
  return doc.documentElement.textContent;
}

function insertTextAtCursor(text, tagword) {
    let promptTextbox = gradioApp().querySelector('#prompt_input > label > textarea');
    let cursorPos = promptTextbox.selectionStart;
    let sanitizedText = acConfig.replaceUnderscores ? text.replaceAll("_", " ") : text;
    let optionalComma = (promptTextbox.value[cursorPos] == ",") ? "" : ", ";

    // Decode the HTML entities in sanitizedText
    sanitizedText = htmlDecode(sanitizedText);

    // Edit prompt text
    var prompt = promptTextbox.value;
    promptTextbox.value = prompt.substring(0, cursorPos - tagword.length) + sanitizedText + optionalComma + prompt.substring(cursorPos);
    prompt = promptTextbox.value += ' ';

    // Update cursor position to after the inserted text
    promptTextbox.selectionStart = cursorPos + sanitizedText.length;
    promptTextbox.selectionEnd = promptTextbox.selectionStart;

    // Hide results after inserting
    hideResults();

    // Update previous tags with the edited prompt to prevent re-searching the same term
    let tags = prompt.match(/[^, ]+/g);
    previousTags = tags;
}


const colors_dark = ["lightblue", "indianred", "unused", "violet", "lightgreen", "orange"];
const colors_light = ["dodgerblue", "firebrick", "unused", "darkorchid", "darkgreen", "darkorange" ]
function addResultsToList(results, tagword) {
    let resultsList = gradioApp().querySelector('#autocompleteResultsList');
    resultsList.innerHTML = "";

    let colors = gradioApp().querySelector('.dark') ? colors_dark : colors_light;

    for (let i = 0; i < results.length; i++) {
        let result = results[i];
        let li = document.createElement("li");
        li.innerHTML = result[0];
        li.style = `color: ${colors[result[1]]};`;
        li.addEventListener("click", function() { insertTextAtCursor(result[0], tagword); });
        resultsList.appendChild(li);
    }
}

function updateSelectionStyle(num) {
    let resultsList = gradioApp().querySelector('#autocompleteResultsList');
    let items = resultsList.getElementsByTagName('li');

    for (let i = 0; i < items.length; i++) {
        items[i].classList.remove('selected');
    }

    items[num].classList.add('selected');
}

allTags = [];
previousTags = [];
results = [];
tagword = "";
resultCount = 0;
function autocomplete(prompt) {
    // Guard for empty prompt
    if (prompt.length == 0) {
        hideResults();
        return;
    }

    // Match tags with RegEx to get the last edited one
    let tags = prompt.match(/[^, ]+/g);
    let diff = difference(tags, previousTags)
    previousTags = tags;

    // Guard for no difference / only whitespace remaining
    if (diff == undefined || diff.length == 0) {
        hideResults();
        return;
    }

    tagword = diff[0]

    // Guard for empty tagword
    if (tagword == undefined || tagword.length == 0) {
        hideResults();
        return;
    }

    // Special case for "<"
    if (tagword === "<") {
        results = allTags.slice(0, acConfig.maxResults);
    } else {
        results = allTags.filter(x => x[0].includes(tagword)).slice(0, acConfig.maxResults);
    }

    resultCount = results.length;

    // Guard for empty results
    if (resultCount == 0) {
        hideResults();
        return;
    }

    showResults();
    addResultsToList(results, tagword);
}


function navigateInList(event) {
    validKeys = ["ArrowUp", "ArrowDown", "Enter", "Escape"];

    if (!validKeys.includes(event.key)) return;
    if (!isVisible) return

    switch (event.key) {
        case "ArrowUp":
            if (selectedTag == null) {
                selectedTag = resultCount - 1;
            } else {
                selectedTag = (selectedTag - 1 + resultCount) % resultCount;
            }
            break;
        case "ArrowDown":
            if (selectedTag == null) {
                selectedTag = 0;
            } else {
                selectedTag = (selectedTag + 1) % resultCount;
            }
            break;
        case "Enter":
            if (selectedTag != null) {
                insertTextAtCursor(results[selectedTag][0], tagword);
            }
            break;
        case "Escape":
            hideResults();
            break;
    }
    // Update highlighting
    if (selectedTag != null)
        updateSelectionStyle(selectedTag);

    // Prevent default behavior
    event.preventDefault();
    event.stopPropagation();
}

onUiUpdate(function(){
    // One-time CSV setup
    if (acConfig == null) acConfig = JSON.parse('{\n	"tagFile": "danbooru.csv",\n	"maxResults": 10,\n	"replaceUnderscores": false}');

    let hidden_tags = gradioApp().querySelector('#hidden_tags > label > textarea');
    let tags_string = hidden_tags.value;
    allTags = parseCSV(tags_string);

    console.log(allTags)

	let promptTextbox = gradioApp().querySelector('#prompt_input > label > textarea');

    if (allTags == null) return;
    if (promptTextbox == null) return;
    if (gradioApp().querySelector('#autocompleteResults') != null) return;

    // Only add listeners once
    if (!promptTextbox.classList.contains('autocomplete')) {
        // Add our new element
        var resultsDiv = gradioApp().querySelector('#autocompleteResults') ?? createResultsDiv();
        promptTextbox.parentNode.insertBefore(resultsDiv, promptTextbox.nextSibling);
        // Hide by default so it doesn't show up on page load
        hideResults();

        // Add autocomplete event listener
        promptTextbox.addEventListener('input', debounce(() => autocomplete(promptTextbox.value), 100));
        // Add focusout event listener
        promptTextbox.addEventListener('focusout', debounce(() => hideResults(), 400));
        // Add up and down arrow event listener
        promptTextbox.addEventListener('keydown', function(e) { navigateInList(e); });


        // Add class so we know we've already added the listeners
        promptTextbox.classList.add('autocomplete');

        // Add style to dom
        let acStyle = document.createElement('style');

        let css = autocompleteCSS_dark;
        if (acStyle.styleSheet) {
            acStyle.styleSheet.cssText = css;
        } else {
            acStyle.appendChild(document.createTextNode(css));
        }
        gradioApp().appendChild(acStyle);
    }
});