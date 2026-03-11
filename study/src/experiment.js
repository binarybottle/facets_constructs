/* Teacher Quality Assessment Study -- Forced-choice pairs */
import { initJsPsych } from 'jspsych';
import htmlButtonResponse from '@jspsych/plugin-html-button-response';
import 'jspsych/css/jspsych.css';

// Initialize jsPsych
const jsPsych = initJsPsych();

// Experiment configuration
let experimentConfig = {
  numTrials: 100,  // Number of forced-choice pairs to present
  itemsFile: 'data/items.csv',  // CSV with Items, Synonym 1, Synonym 2, Synonym 3
  // Attention checks: ~1 every 5 min if ~60 screens between checks
  attentionCheckPairs: [
    ['Paying attention', 'Continuing to stop'],
    ['Respect for others ', 'Reading the drivel'],
    ['Playing with others', 'Receptive unfamiliar'],
    ['Managing emotions', 'Learning the fatigue'],
    ['Social skills', 'Learning unuseless'],
  ],  // Each pair is [correctWord, distractorWord]; one pair is chosen at random per check
  screensBetweenAttentionChecks: 60,  // Number of main (forced-choice) trials between attention checks
};

// OSF and server configuration
const osfNodeId = "dcv5z";

// Prolific completion URL with placeholder for the completion code
const PROLIFIC_COMPLETION_URL = "https://app.prolific.co/submissions/complete?cc=";
const COMPLETION_CODE = "C1NWOX09";
const NO_CONSENT_CODE = "C1O763HF";

// Get Prolific ID from URL parameters
let prolificID = '';
function getUrlParam(name) {
  const urlParams = new URLSearchParams(window.location.search);
  return urlParams.get(name);
}
prolificID = getUrlParam('PROLIFIC_PID') || 'unknown';
if (!prolificID || prolificID === 'unknown') {
  console.warn("Warning: Prolific ID is not available. Proceeding with 'unknown' ID.");
}

// Function to redirect to Prolific
function redirectToProlific(code) {
  console.log(`Redirecting to Prolific with code: ${code}`);
  window.location.href = PROLIFIC_COMPLETION_URL + code;
}

// Load OSF API token
async function loadOSFToken() {
  try {
    const response = await fetch('./token.json');
    const data = await response.json();
    console.log('OSF API token loaded');
    return data.osf_api_token;
  } catch (error) {
    console.error('Error loading OSF token:', error);
    return null;
  }
}

// Load items from CSV file
async function loadItems(filename) {
  try {
    const response = await fetch(`./${filename}`);
    const csvText = await response.text();
    const lines = csvText.split('\n').map(row => row.trim()).filter(row => row);
    
    // Skip header row
    const dataLines = lines.slice(1);
    
    const items = [];
    const allTerms = []; // All items + all synonyms for random selection
    
    dataLines.forEach((line, index) => {
      // Parse CSV properly handling quoted fields
      const cols = line.match(/(".*?"|[^,]+)(?=\s*,|\s*$)/g) || [];
      const cleanCols = cols.map(c => c.replace(/^"(.*)"$/, '$1').trim());
      
      const item = cleanCols[0] || '';
      const syn1 = cleanCols[1] || '';
      const syn2 = cleanCols[2] || '';
      const syn3 = cleanCols[3] || '';
      
      if (item) {
        items.push({
          name: item,
          synonyms: [syn1, syn2, syn3].filter(s => s)
        });
        
        // Add item and all its synonyms to the pool
        allTerms.push({ term: item, isItem: true, sourceItem: item });
        if (syn1) allTerms.push({ term: syn1, isItem: false, sourceItem: item });
        if (syn2) allTerms.push({ term: syn2, isItem: false, sourceItem: item });
        if (syn3) allTerms.push({ term: syn3, isItem: false, sourceItem: item });
      }
    });
    
    console.log(`Loaded ${items.length} items with ${allTerms.length} total terms`);
    return { items, allTerms };
  } catch (error) {
    console.error(`Error loading ${filename}:`, error);
    return { items: [], allTerms: [] };
  }
}

// Generate random pairs for the experiment
function generateRandomPairs(items, allTerms, numPairs) {
  const pairs = [];
  const usedPairs = new Set();
  
  for (let i = 0; i < numPairs; i++) {
    let attempts = 0;
    let validPair = false;
    
    while (!validPair && attempts < 100) {
      attempts++;
      
      // Pick a random item
      const item1 = items[Math.floor(Math.random() * items.length)];
      
      // Pick a random term (could be another item or any synonym, including from same source)
      // Just exclude the exact same term
      const eligibleTerms = allTerms.filter(t => t.term !== item1.name);
      if (eligibleTerms.length === 0) continue;
      
      const term2 = eligibleTerms[Math.floor(Math.random() * eligibleTerms.length)];
      
      // Create a unique key for this pair (order-independent)
      const pairKey = [item1.name, term2.term].sort().join('|||');
      
      if (!usedPairs.has(pairKey)) {
        usedPairs.add(pairKey);
        
        // Randomize left/right position
        if (Math.random() < 0.5) {
          pairs.push({
            left: { term: item1.name, isItem: true, sourceItem: item1.name },
            right: term2
          });
        } else {
          pairs.push({
            left: term2,
            right: { term: item1.name, isItem: true, sourceItem: item1.name }
          });
        }
        validPair = true;
      }
    }
  }
  
  return pairs;
}

// Function to create and update a progress counter
function createProgressCounter() {
  const existingCounter = document.getElementById('progress-counter');
  if (existingCounter) {
    existingCounter.remove();
  }
  
  const counterContainer = document.createElement('div');
  counterContainer.id = 'progress-counter';
  counterContainer.style = `
    position: fixed;
    top: 20px;
    right: 20px;
    background-color: rgba(240, 240, 240, 0.9);
    color: #333;
    border-radius: 8px;
    padding: 8px 15px;
    font-size: 16px;
    font-weight: bold;
    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    z-index: 9999;
    display: block;
  `;
  document.body.appendChild(counterContainer);
}

function updateProgressCounter(current, total) {
  const counterContainer = document.getElementById('progress-counter');
  if (counterContainer) {
    counterContainer.textContent = `${current} of ${total}`;
  }
}

function hideProgressCounter() {
  const counterContainer = document.getElementById('progress-counter');
  if (counterContainer) {
    counterContainer.style.display = 'none';
  }
}

function showProgressCounter() {
  const counterContainer = document.getElementById('progress-counter');
  if (counterContainer) {
    counterContainer.style.display = 'block';
  } else {
    createProgressCounter();
  }
}

// Global styles
function setGlobalStyles() {
  const style = document.createElement('style');
  style.textContent = `
    .jspsych-content {
      max-width: 90% !important;
      font-size: 20px !important;
    }
    .jspsych-btn {
      font-size: 18px !important;
      padding: 15px 30px !important;
      margin: 15px !important;
      min-width: 200px;
    }
    .choice-btn {
      font-size: 22px !important;
      padding: 20px 40px !important;
      margin: 20px !important;
      min-width: 250px;
      background-color: #f0f0f0;
      border: 2px solid #ccc;
      border-radius: 10px;
      cursor: pointer;
      transition: all 0.2s ease;
    }
    .choice-btn:hover {
      background-color: #e0e0e0;
      border-color: #999;
      transform: scale(1.02);
    }
    .prompt-text {
      font-size: 24px;
      margin-bottom: 40px;
      color: #333;
      line-height: 1.5;
    }
    .vs-text {
      font-size: 20px;
      color: #666;
      margin: 0 20px;
    }
    .choice-container {
      display: flex;
      justify-content: center;
      align-items: center;
      flex-wrap: wrap;
      gap: 20px;
      margin-top: 30px;
    }
    .thank-you-container {
      text-align: center;
      max-width: 800px;
      margin: 0 auto;
      padding: 30px;
      background-color: #f9f9f9;
      border-radius: 15px;
      box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .thank-you-title {
      font-size: 32px;
      color: #2c3e50;
      margin-bottom: 20px;
    }
    .thank-you-message {
      font-size: 20px;
      color: #34495e;
      line-height: 1.6;
      margin-bottom: 30px;
    }
    .thank-you-button {
      font-size: 22px !important;
      padding: 15px 30px !important;
      background-color: #3498db;
      color: white;
      border: none;
      border-radius: 8px;
      cursor: pointer;
      transition: background-color 0.3s;
    }
    .thank-you-button:hover {
      background-color: #2980b9;
    }
    @keyframes checkmark {
      0% { transform: scale(0); opacity: 0; }
      50% { transform: scale(1.2); opacity: 1; }
      100% { transform: scale(1); opacity: 1; }
    }
    .checkmark {
      display: inline-block;
      width: 80px;
      height: 80px;
      border-radius: 50%;
      background-color: #2ecc71;
      margin-bottom: 20px;
      position: relative;
      animation: checkmark 0.5s ease-in-out forwards;
    }
    .checkmark:after {
      content: '';
      position: absolute;
      top: 45%;
      left: 30%;
      width: 35%;
      height: 15%;
      border-left: 4px solid white;
      border-bottom: 4px solid white;
      transform: rotate(-45deg);
    }
    /* Demographics survey styles */
    .survey-form {
      text-align: left;
      max-width: 660px;
      margin: 0 auto;
      font-size: 16px;
      padding: 0 20px 10px;
    }
    .survey-form h2 {
      text-align: center;
      font-size: 24px;
      margin-bottom: 6px;
      color: #2c3e50;
    }
    .survey-subtitle {
      text-align: center;
      color: #666;
      font-size: 14px;
      margin-bottom: 28px;
    }
    .survey-q {
      margin-bottom: 22px;
      padding-bottom: 18px;
      border-bottom: 1px solid #e8e8e8;
    }
    .survey-q:last-of-type { border-bottom: none; }
    .survey-q-label {
      font-weight: bold;
      color: #2c3e50;
      margin-bottom: 10px;
      line-height: 1.4;
    }
    .survey-hint {
      font-weight: normal;
      font-style: italic;
      color: #888;
      font-size: 13px;
    }
    .survey-options {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .survey-options-row {
      flex-direction: row;
      flex-wrap: wrap;
      gap: 12px 28px;
    }
    .survey-option {
      display: flex;
      align-items: center;
      gap: 8px;
      cursor: pointer;
      font-size: 15px;
      color: #333;
    }
    .survey-option input { cursor: pointer; width: 16px; height: 16px; }
    .survey-select {
      width: 100%;
      padding: 9px 12px;
      font-size: 15px;
      border: 1px solid #ccc;
      border-radius: 6px;
      background: #fff;
      color: #333;
    }
    .survey-required-note {
      text-align: center;
      font-size: 13px;
      color: #999;
      margin-top: 12px;
    }
  `;
  document.head.appendChild(style);
}

// Country list for demographics survey
const COUNTRIES = [
  "Afghanistan","Albania","Algeria","Andorra","Angola","Antigua and Barbuda","Argentina",
  "Armenia","Australia","Austria","Azerbaijan","Bahamas","Bahrain","Bangladesh","Barbados",
  "Belarus","Belgium","Belize","Benin","Bhutan","Bolivia","Bosnia and Herzegovina","Botswana",
  "Brazil","Brunei","Bulgaria","Burkina Faso","Burundi","Cabo Verde","Cambodia","Cameroon",
  "Canada","Central African Republic","Chad","Chile","China","Colombia","Comoros","Congo",
  "Costa Rica","Croatia","Cuba","Cyprus","Czech Republic","Democratic Republic of the Congo",
  "Denmark","Djibouti","Dominica","Dominican Republic","Ecuador","Egypt","El Salvador",
  "Equatorial Guinea","Eritrea","Estonia","Eswatini","Ethiopia","Fiji","Finland","France",
  "Gabon","Gambia","Georgia","Germany","Ghana","Greece","Grenada","Guatemala","Guinea",
  "Guinea-Bissau","Guyana","Haiti","Honduras","Hungary","Iceland","India","Indonesia","Iran",
  "Iraq","Ireland","Israel","Italy","Jamaica","Japan","Jordan","Kazakhstan","Kenya","Kiribati",
  "Kuwait","Kyrgyzstan","Laos","Latvia","Lebanon","Lesotho","Liberia","Libya","Liechtenstein",
  "Lithuania","Luxembourg","Madagascar","Malawi","Malaysia","Maldives","Mali","Malta",
  "Marshall Islands","Mauritania","Mauritius","Mexico","Micronesia","Moldova","Monaco",
  "Mongolia","Montenegro","Morocco","Mozambique","Myanmar","Namibia","Nauru","Nepal",
  "Netherlands","New Zealand","Nicaragua","Niger","Nigeria","North Korea","North Macedonia",
  "Norway","Oman","Pakistan","Palau","Palestine","Panama","Papua New Guinea","Paraguay",
  "Peru","Philippines","Poland","Portugal","Qatar","Romania","Russia","Rwanda",
  "Saint Kitts and Nevis","Saint Lucia","Saint Vincent and the Grenadines","Samoa",
  "San Marino","Sao Tome and Principe","Saudi Arabia","Senegal","Serbia","Seychelles",
  "Sierra Leone","Singapore","Slovakia","Slovenia","Solomon Islands","Somalia","South Africa",
  "South Korea","South Sudan","Spain","Sri Lanka","Sudan","Suriname","Sweden","Switzerland",
  "Syria","Taiwan","Tajikistan","Tanzania","Thailand","Timor-Leste","Togo","Tonga",
  "Trinidad and Tobago","Tunisia","Turkey","Turkmenistan","Tuvalu","Uganda","Ukraine",
  "United Arab Emirates","United Kingdom","United States","Uruguay","Uzbekistan","Vanuatu",
  "Vatican City","Venezuela","Vietnam","Yemen","Zambia","Zimbabwe"
];

// Demographics survey (runs once, right after consent)
function createDemographicsTrial() {
  const countryOptions = COUNTRIES.map(c => `<option value="${c}">${c}</option>`).join('');

  // Closure variable: populated on every form change so on_finish can read it
  // after jsPsych clears the DOM (which happens before on_finish is called).
  const formData = {
    grade_levels: '', special_needs: '', school_type: '',
    experience: '', class_size: '', country: ''
  };

  return {
    type: htmlButtonResponse,
    stimulus: `
      <div class="survey-form">
        <h2>About Your Teaching Background</h2>
        <p class="survey-subtitle">Please answer the following questions before beginning the main task.<br>All fields are required.</p>

        <div class="survey-q">
          <div class="survey-q-label">1. Which grade levels are you actively teaching? <span class="survey-hint">(select all that apply)</span></div>
          <div class="survey-options">
            <label class="survey-option"><input type="checkbox" name="grade_level" value="kindergarten"> Kindergarten</label>
            <label class="survey-option"><input type="checkbox" name="grade_level" value="primary"> Primary school (grades 1–5 or 6)</label>
            <label class="survey-option"><input type="checkbox" name="grade_level" value="middle"> Middle school (grades 6 or 7–8)</label>
            <label class="survey-option"><input type="checkbox" name="grade_level" value="high_school"> High school (grades 9–12)</label>
            <label class="survey-option"><input type="checkbox" name="grade_level" value="vocational"> Vocational / Trade school</label>
          </div>
        </div>

        <div class="survey-q">
          <div class="survey-q-label">2. Are you teaching specifically students with special needs?</div>
          <div class="survey-options survey-options-row">
            <label class="survey-option"><input type="radio" name="special_needs" value="yes"> Yes</label>
            <label class="survey-option"><input type="radio" name="special_needs" value="no"> No</label>
          </div>
        </div>

        <div class="survey-q">
          <div class="survey-q-label">3. What type of school(s) do you teach at?</div>
          <div class="survey-options survey-options-row">
            <label class="survey-option"><input type="checkbox" name="school_type" value="public"> Public</label>
            <label class="survey-option"><input type="checkbox" name="school_type" value="private"> Private</label>
          </div>
        </div>

        <div class="survey-q">
          <div class="survey-q-label">4. How many years of teaching experience do you have?</div>
          <div class="survey-options survey-options-row">
            <label class="survey-option"><input type="radio" name="experience" value="1"> 1 year</label>
            <label class="survey-option"><input type="radio" name="experience" value="2-5"> 2–5 years</label>
            <label class="survey-option"><input type="radio" name="experience" value="6-10"> 6-10 years</label>
            <label class="survey-option"><input type="radio" name="experience" value=">10"> More than 10 years</label>
          </div>
        </div>

        <div class="survey-q">
          <div class="survey-q-label">5. What is the average number of students in your current classroom(s)?</div>
          <div class="survey-options survey-options-row">
            <label class="survey-option"><input type="radio" name="class_size" value="<15"> Fewer than 15</label>
            <label class="survey-option"><input type="radio" name="class_size" value="15-25"> 15–25</label>
            <label class="survey-option"><input type="radio" name="class_size" value=">25"> More than 25</label>
          </div>
        </div>

        <div class="survey-q">
          <div class="survey-q-label">6. Which country do you work in?</div>
          <select id="country-select" class="survey-select">
            <option value="">— Select a country —</option>
            ${countryOptions}
          </select>
        </div>

        <p class="survey-required-note">Please answer all questions to continue.</p>
      </div>
    `,
    choices: ["Continue"],
    button_html: '<button class="jspsych-btn" id="survey-submit" style="font-size: 18px; padding: 12px 36px; margin-top: 10px;">%choice%</button>',
    on_load: function() {
      hideProgressCounter();

      // jsPsych attaches its click handler (bubble phase) to the wrapper div.
      // It never checks `disabled`, so we must intercept in the capture phase,
      // which fires before the bubble phase, to block premature submission.
      const wrapper = document.getElementById('jspsych-html-button-response-button-0');
      const btn = wrapper ? wrapper.querySelector('button') : null;

      function isValid() {
        return formData.grade_levels !== '' && formData.special_needs !== '' &&
               formData.school_type  !== '' && formData.experience    !== '' &&
               formData.class_size   !== '' && formData.country       !== '';
      }

      // Block jsPsych's click handler when form is incomplete
      if (wrapper) {
        wrapper.addEventListener('click', function(e) {
          if (!isValid()) {
            e.stopImmediatePropagation();
            e.preventDefault();
          }
        }, true); // capture phase — fires before jsPsych's bubble-phase handler
      }

      // Visual state of the button
      function updateButtonStyle() {
        if (!btn) return;
        btn.style.opacity = isValid() ? '1' : '0.4';
        btn.style.cursor  = isValid() ? 'pointer' : 'not-allowed';
      }
      if (btn) { btn.style.opacity = '0.4'; btn.style.cursor = 'not-allowed'; }

      function collectAndCheck() {
        formData.grade_levels  = Array.from(document.querySelectorAll('input[name="grade_level"]:checked'))
                                      .map(el => el.value).join('|');
        formData.special_needs = document.querySelector('input[name="special_needs"]:checked')?.value || '';
        formData.school_type   = document.querySelector('input[name="school_type"]:checked')?.value  || '';
        formData.experience    = document.querySelector('input[name="experience"]:checked')?.value   || '';
        formData.class_size    = document.querySelector('input[name="class_size"]:checked')?.value   || '';
        formData.country       = document.getElementById('country-select')?.value                   || '';
        updateButtonStyle();
      }

      document.querySelectorAll('input[type="checkbox"], input[type="radio"]')
        .forEach(el => el.addEventListener('change', collectAndCheck));
      document.getElementById('country-select').addEventListener('change', collectAndCheck);
    },
    on_finish: function(data) {
      // DOM is already cleared by jsPsych at this point — read from closure instead
      data.task = 'demographics';
      Object.assign(data, formData);
    }
  };
}

// Consent trial for teacher participants
const consentTrial = {
  type: htmlButtonResponse,
  stimulus: `
    <div class='instruction' style='text-align: left; max-width: 700px; margin: 0 auto; font-size: 14px;'> 
      <h2 style='text-align: center; font-size: 22px;'>Welcome</h2>
      <dl style='line-height: 1.4;'>
          <dt style='font-weight: bold; margin-top: 10px;'>Purpose</dt>
          <dd style='margin-left: 0; margin-bottom: 8px;'>This study is designed to understand which factors K-12 teachers consider to be the most important aspects of their students’ development and functioning.  Your responses will help inform the development of educational assessment tools.</dd>

          <dt style='font-weight: bold; margin-top: 10px;'>Eligibility</dt>
          <dd style='margin-left: 0; margin-bottom: 8px;'>You must be a <b>current K-12 teacher</b> to participate in this study.</dd>

          <dt style='font-weight: bold; margin-top: 10px;'>Procedures</dt>
          <dd style='margin-left: 0; margin-bottom: 8px;'>Before the main task, you will be asked a few brief background questions about your teaching experience. Then you will be presented with pairs of student qualities (such as "Self-Control" vs. "Empathy") and asked to choose which of the two you consider to be a more relevant and important characteristic of students with whom you work. We value your experience and are interested in your considered professional judgment about each pair.</dd>

          <dt style='font-weight: bold; margin-top: 10px;'>Duration</dt>
          <dd style='margin-left: 0; margin-bottom: 8px;'>This study takes approximately 5-10 minutes to complete.</dd>

                    <dt style='font-weight: bold; margin-top: 10px;'>Risks</dt>
          <dd style='margin-left: 0; margin-bottom: 8px;'>There are no anticipated risks or discomforts from this research.</dd>

          <dt style='font-weight: bold; margin-top: 10px;'>Benefits</dt>
          <dd style='margin-left: 0; margin-bottom: 8px;'>Your participation will contribute to improving how we understand and assess important elements of youth development, function, and personal qualities.</dd>

          <dt style='font-weight: bold; margin-top: 10px;'>Compensation</dt>
          <dd style='margin-left: 0; margin-bottom: 8px;'>If you decide to participate, you will be compensated for your participation as described in the Prolific study listing.</dd>

          <dt style='font-weight: bold; margin-top: 10px;'>Participation</dt>
          <dd style='margin-left: 0; margin-bottom: 8px;'>Taking part in this research study is your decision. You can decide to participate and then change your mind at any point.</dd>

          <dt style='font-weight: bold; margin-top: 10px;'>Contact</dt>
          <dd style='margin-left: 0; margin-bottom: 8px;'>If you have any questions about the purpose, procedures, or any other issues related to this research study you may contact the Principal Investigator, Dr. Arno Klein, at arno.klein@childmind.org.</dd>
      </dl>
      <p style='text-align: center; font-weight: bold; margin-top: 15px; font-size: 15px;'>
        Do you consent to participate in this study? <br><span style='font-weight: normal;'>You must be 18 years of age or older to participate.</span>
      </p>
    </div>
  `,
  choices: ["I consent", "I do not consent"],
  button_html: '<button class="jspsych-btn" style="font-size: 14px; padding: 8px 18px; margin: 0 10px;">%choice%</button>',
  on_load: function() {
    hideProgressCounter();
  },
  on_finish: function(data) {
    if (data.response === 1) {  // "I do not consent"
      redirectToProlific(NO_CONSENT_CODE);
    }
  }
};

// Create a forced-choice trial
function createChoiceTrial(pair, trialIndex, totalTrials) {
  return {
    type: htmlButtonResponse,
    stimulus: `
      <div style="text-align: center;">
        <p class="prompt-text">
          When considering your students,<br>
          which of the two presented qualities<br>
          is more relevant and important<br>
          for you to understand or assess?<br>
          <br>
          Please use your professional judgment.
        </p>
      </div>
    `,
    choices: [pair.left.term, pair.right.term],
    button_html: '<button class="choice-btn">%choice%</button>',
    on_load: function() {
      showProgressCounter();
      updateProgressCounter(trialIndex + 1, totalTrials);
    },
    on_finish: function(data) {
      const chosenIndex = data.response;
      const chosen = chosenIndex === 0 ? pair.left : pair.right;
      const unchosen = chosenIndex === 0 ? pair.right : pair.left;
      
      data.task = 'forced_choice';
      data.trial_index = trialIndex;
      data.left_term = pair.left.term;
      data.left_is_item = pair.left.isItem;
      data.left_source = pair.left.sourceItem;
      data.right_term = pair.right.term;
      data.right_is_item = pair.right.isItem;
      data.right_source = pair.right.sourceItem;
      data.chosen_term = chosen.term;
      data.chosen_is_item = chosen.isItem;
      data.chosen_source = chosen.sourceItem;
      data.unchosen_term = unchosen.term;
      data.unchosen_is_item = unchosen.isItem;
      data.unchosen_source = unchosen.sourceItem;
      data.response_time = data.rt;
    }
  };
}

// Create an attention-check trial (pair = [correctWord, distractorWord])
function createAttentionCheckTrial(pair, overallIndex, totalTrials) {
  const [correctWord, distractorWord] = pair;
  const leftWord = Math.random() < 0.5 ? correctWord : distractorWord;
  const rightWord = leftWord === correctWord ? distractorWord : correctWord;
  const correctChoiceIndex = leftWord === correctWord ? 0 : 1;

  return {
    type: htmlButtonResponse,
    stimulus: `
      <div style="text-align: center;">
        <p class="prompt-text">
          To show you are paying attention,<br>
          please select the word <strong>\"${correctWord}\"</strong> below.
        </p>
      </div>
    `,
    choices: [leftWord, rightWord],
    button_html: '<button class="choice-btn">%choice%</button>',
    on_load: function() {
      showProgressCounter();
      updateProgressCounter(overallIndex + 1, totalTrials);
    },
    on_finish: function(data) {
      const passed = data.response === correctChoiceIndex;
      data.task = 'attention_check';
      data.trial_index = overallIndex;
      data.correct_word = correctWord;
      data.chosen_word = data.response === 0 ? leftWord : rightWord;
      data.passed = passed;
      data.response_time = data.rt;
    }
  };
}

// Thank you trial
const thankYouTrial = {
  type: htmlButtonResponse,
  stimulus: `
    <div class="thank-you-container">
      <div class="checkmark"></div>
      <h2 class="thank-you-title">Thank You!</h2>
      <p class="thank-you-message">
        Thank you for your participation in this study!<br>
        Your responses will help us understand what qualities teachers value in their students.<br><br>
        All of your data has been successfully recorded.
      </p>
    </div>
  `,
  choices: ["Complete Study"],
  button_html: '<button class="thank-you-button">%choice%</button>',
  on_load: function() {
    hideProgressCounter();
  },
  on_finish: function() {
    endExperiment();
  }
};

// Convert data to CSV format
function convertToCSV(data) {
  const headers = [
    'user_id', 'task', 'trial_index',
    'left_term', 'left_is_item', 'left_source',
    'right_term', 'right_is_item', 'right_source',
    'chosen_term', 'chosen_is_item', 'chosen_source',
    'unchosen_term', 'unchosen_is_item', 'unchosen_source',
    'response_time',
    'correct_word', 'chosen_word', 'passed',
    'grade_levels', 'special_needs', 'school_type', 'experience', 'class_size', 'country'
  ];

  const q = s => `"${(s || '').toString().replace(/"/g, '""')}"`;

  let content = headers.join(',') + '\n';

  data.forEach(trial => {
    if (trial.task === 'forced_choice') {
      const row = [
        prolificID, 'forced_choice', trial.trial_index,
        q(trial.left_term), trial.left_is_item, q(trial.left_source),
        q(trial.right_term), trial.right_is_item, q(trial.right_source),
        q(trial.chosen_term), trial.chosen_is_item, q(trial.chosen_source),
        q(trial.unchosen_term), trial.unchosen_is_item, q(trial.unchosen_source),
        trial.response_time,
        '', '', '',
        '', '', '', '', '', ''
      ];
      content += row.join(',') + '\n';
    } else if (trial.task === 'attention_check') {
      const row = [
        prolificID, 'attention_check', trial.trial_index,
        '', '', '', '', '', '',
        '', '', '', '', '', '',
        trial.response_time,
        q(trial.correct_word), q(trial.chosen_word), trial.passed,
        '', '', '', '', '', ''
      ];
      content += row.join(',') + '\n';
    } else if (trial.task === 'demographics') {
      const row = [
        prolificID, 'demographics', '',
        '', '', '', '', '', '',
        '', '', '', '', '', '',
        '',
        '', '', '',
        q(trial.grade_levels), q(trial.special_needs), q(trial.school_type),
        q(trial.experience), q(trial.class_size), q(trial.country)
      ];
      content += row.join(',') + '\n';
    }
  });

  return content;
}

// Store data locally
function storeDataLocally(content, prolificID) {
  try {
    const timestamp = Date.now();
    const fileName = `choice_data_${prolificID}_${timestamp}.csv`;
    localStorage.setItem(fileName, content);
    console.log('Data saved locally:', fileName);
  } catch (error) {
    console.error('Error saving data locally:', error);
  }
}

// Upload to OSF
async function uploadToOSF(url, data, token) {
  const response = await fetch(url, {
    method: 'PUT',
    headers: {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'text/csv',
    },
    body: data,
  });

  if (!response.ok) {
    const errorDetails = await response.text();
    throw new Error(`Upload error! Status: ${response.status}, Details: ${errorDetails}`);
  }
}

async function uploadWithRetry(url, data, token, maxRetries = 3) {
  let attempt = 0;
  while (attempt < maxRetries) {
    try {
      await uploadToOSF(url, data, token);
      console.log(`Upload successful after ${attempt + 1} attempt(s)`);
      break;
    } catch (error) {
      console.error(`Attempt ${attempt + 1} failed:`, error);
      attempt++;
      if (attempt >= maxRetries) {
        throw new Error(`Failed to upload after ${maxRetries} attempts`);
      }
      await new Promise(resolve => setTimeout(resolve, 2000));
    }
  }
}

async function storeDataOnOSF(data) {
  const osfToken = await loadOSFToken();
  if (!osfToken) {
    console.error('OSF API token not available');
    return;
  }

  const csvContent = convertToCSV(data);
  const dataUrl = `https://files.osf.io/v1/resources/${osfNodeId}/providers/osfstorage/?kind=file&name=choice_data_${prolificID}_${Date.now()}.csv`;

  try {
    await uploadWithRetry(dataUrl, csvContent, osfToken);
    storeDataLocally(csvContent, prolificID);
    console.log('Data successfully stored on OSF');
  } catch (error) {
    console.error('Error storing data:', error);
    storeDataLocally(csvContent, prolificID);
  }
}

// End experiment
function endExperiment() {
  const experimentData = jsPsych.data.get().values();
  const validData = experimentData.filter(trial =>
    trial.task === 'forced_choice' || trial.task === 'attention_check' || trial.task === 'demographics'
  );

  storeDataOnOSF(validData)
    .then(() => console.log('Data stored successfully'))
    .catch(error => console.error('Error storing data:', error))
    .finally(() => redirectToProlific(COMPLETION_CODE));
}

// Main experiment function
async function runExperiment(options = {}) {
  Object.assign(experimentConfig, options);
  
  setGlobalStyles();
  createProgressCounter();
  hideProgressCounter();
  
  // Load items
  const { items, allTerms } = await loadItems(experimentConfig.itemsFile);
  
  if (items.length === 0) {
    jsPsych.endExperiment('Error loading items');
    return;
  }
  
  // Generate random pairs
  const pairs = generateRandomPairs(items, allTerms, experimentConfig.numTrials);
  console.log(`Generated ${pairs.length} pairs for experiment`);

  const attentionCheckPairs = experimentConfig.attentionCheckPairs || [];
  const screensBetween = experimentConfig.screensBetweenAttentionChecks ?? 60;
  const numAttentionChecks = attentionCheckPairs.length > 0 && screensBetween > 0
    ? Math.floor(pairs.length / screensBetween)
    : 0;
  const totalTrials = pairs.length + numAttentionChecks;

  // Build interleaved trial list: main trials with an attention check every screensBetween
  const mainAndAttentionTrials = [];
  let mainIndex = 0;
  for (let i = 0; i < numAttentionChecks; i++) {
    for (let j = 0; j < screensBetween; j++) {
      mainAndAttentionTrials.push(createChoiceTrial(pairs[mainIndex], mainIndex, totalTrials));
      mainIndex++;
    }
    const overallIndex = mainAndAttentionTrials.length;
    const acPair = attentionCheckPairs[Math.floor(Math.random() * attentionCheckPairs.length)];
    mainAndAttentionTrials.push(createAttentionCheckTrial(acPair, overallIndex, totalTrials));
  }
  while (mainIndex < pairs.length) {
    mainAndAttentionTrials.push(createChoiceTrial(pairs[mainIndex], mainIndex, totalTrials));
    mainIndex++;
  }
  // Fix trial_index and total for progress: use position in combined timeline
  mainAndAttentionTrials.forEach((trial, idx) => {
    const onLoad = trial.on_load;
    trial.on_load = function() {
      if (onLoad) onLoad.call(this);
      const counterEl = document.getElementById('progress-counter');
      if (counterEl) counterEl.textContent = `${idx + 1} of ${totalTrials}`;
    };
  });

  const timeline = [];

  // Add consent
  timeline.push(consentTrial);

  // Create main experiment timeline (only runs if consent given)
  const experimentTimeline = {
    timeline: [
      // Background questions
      createDemographicsTrial(),
      // Instructions
      {
        type: htmlButtonResponse,
        stimulus: `
          <div style="text-align: center; max-width: 700px; margin: 0 auto;">
            <h2>Instructions</h2>
            <p style="font-size: 20px; line-height: 1.6;">
              You will see pairs of student qualities.<br><br>
              For each pair, choose the quality that is <strong>more relevant and important</strong>
              for you to understand when reflecting on your students.
            </p>
          </div>
        `,
        choices: ["Begin"],
        button_html: '<button class="jspsych-btn" style="font-size: 20px; padding: 15px 40px;">%choice%</button>',
        on_load: hideProgressCounter
      },
      // Choice trials and attention checks interleaved
      ...mainAndAttentionTrials,
      // Thank you
      thankYouTrial
    ],
    conditional_function: function() {
      return jsPsych.data.get().last(1).values()[0].response === 0;
    }
  };

  timeline.push(experimentTimeline);

  jsPsych.run(timeline);
}

// Start the experiment
runExperiment({
  numTrials: experimentConfig.numTrials,
  itemsFile: experimentConfig.itemsFile
});
