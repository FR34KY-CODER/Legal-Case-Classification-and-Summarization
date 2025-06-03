#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Importing necessary libraries

import pandas as pd

ik = pd.read_csv(r'C:\Users\FR34K\Desktop\Coding\BTP\Legal_Case_Dataset_Final.csv')

# Specify the columns to combine in the desired order
columns_to_combine = ["Fact", "Issue", "Petitioner's Argument", "Respondent's Argument",
    "Precedent Analysis", "Analysis of the law", "Court's Reasoning", "Conclusion"]

# Create the new column by concatenating the text data row-wise
ik['Case Content'] = ik[columns_to_combine].fillna('').astype(str).agg(''.join, axis=1)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Data Preprocessing
# Importing necessary libraries for text preprocessing

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')

legal_terms = {'herein', 'thereof', 'whereof', 'hereto', 'therein'}
stop_words = set(stopwords.words('english')) - legal_terms


# Adding custom stop words
lemmatizer = WordNetLemmatizer()

def clean_legal_text(text):

    text = text.lower()

    citations = re.findall(r'air\s+\d{4}\s+\w+\s+\d+', text)

    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    text = re.sub(r'\s+', ' ', text).strip()

    sentences = nltk.sent_tokenize(text)

    cleaned_sentences = []
    for sent in sentences:
        tokens = nltk.word_tokenize(sent)
        tokens = [word for word in tokens if word not in stop_words]
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        cleaned_sentences.append(' '.join(tokens))

    cleaned_text = ' '.join(cleaned_sentences)
    if citations:
        cleaned_text += ' ' + ' '.join(citations)
    return cleaned_text

ik['cleaned_text'] = ik['Case Content'].apply(clean_legal_text)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Data Visualization
# Importing necessary libraries for data visualization

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

ik['original_word_count'] = ik['Case Content'].apply(lambda x: len(x.split()))
ik['cleaned_word_count'] = ik['cleaned_text'].apply(lambda x: len(x.split()))

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.histplot(ik['original_word_count'], color='royalblue', kde=True)
plt.title('Original Document Length Distribution')
plt.xlabel('Word Count')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
sns.histplot(ik['cleaned_word_count'], color='royalblue', kde=True)
plt.title('Cleaned Document Length Distribution')
plt.xlabel('Word Count')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

print("Original text length statistics:")
print(ik['original_word_count'].describe())
print("\nCleaned text length statistics:")
print(ik['cleaned_word_count'].describe())

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# N-gram Analysis

from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter

def extract_ngrams(corpus, ngram_range, top_n=20):
    vectorizer = CountVectorizer(ngram_range=ngram_range)
    X = vectorizer.fit_transform(corpus)

    features = vectorizer.get_feature_names_out()
    counts = X.sum(axis=0).A1

    ngrams_df = pd.DataFrame({'ngram': features, 'count': counts})
    ngrams_df = ngrams_df.sort_values('count', ascending=False).head(top_n)

    return ngrams_df

top_unigrams = extract_ngrams(ik['cleaned_text'], (1, 1), 30)
top_bigrams = extract_ngrams(ik['cleaned_text'], (2, 2), 30)
top_trigrams = extract_ngrams(ik['cleaned_text'], (3, 3), 30)

print("Top 30 Legal Terms:")
print(top_unigrams)

print("\nTop 30 Legal Bigrams (Two-word phrases):")
print(top_bigrams)

print("\nTop 30 Legal Trigrams (Three-word phrases):")
print(top_trigrams)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Categorizing legal cases based on keywords

# Define keywords for each category
category_keywords = {
    # LABOUR MATTERS
    'Dismissal': ['termination', 'misconduct', 'disciplinary action', 'show cause notice', 'natural justice violation', 'departmental inquiry', 'charge sheet', 'termination benefits'],
    'Retrenchment': ['retrenchment compensation', 'section 25N', 'layoff', 'closure', 'retrenchment notice', 'industrial disputes act 1947', 'compensation calculation', 'last come first go', 'government permission', 'retrenchment approval'],
    'Contract Labour': ['contract labour regulation', 'principal employer', 'licensing contractor', 'abolition notification', 'sham contract', 'tripartite agreement', 'labour commissioner', 'contract worker benefits', 'section 10', 'direct employment'],
    'Matters relating to wages': ['minimum wages', 'wage revision', 'dearness allowance', 'pay parity', 'arrears calculation', 'wage board', 'equal remuneration', 'basic pay', 'allowances dispute', 'wage regularization'],
    'Workmen Compensation Act': ['employment injury schedule iii','section 3 employer liability','occupational disease notification','disablement percentage assessment','compensation commissioner appeal','dependency benefits calculation','employer liability insurance','accident arising employment','mesothelioma compensation','silicosis diagnosis report'],
    'ESI': ['employees state insurance', 'esi contribution', 'medical benefit', 'disability benefit', 'dependent benefit', 'esi corporation', 'section 46', 'insurable employment', 'registration certificate', 'benefit period'],
    'Factory Act': ['factory license', 'occupier liability', 'hazardous process', 'section 7A', 'annual leave', 'health measures', 'safety officer', 'canteen facilities', 'creche facility', 'dangerous machinery'],
    'Industrial Employment (Standing Order)': ['standing order certification', 'service rules', 'shift working', 'attendance', 'leave rules', 'classification of workers', 'suspension pending inquiry', 'holiday list', 'termination procedure', 'grievance redressal'],
    'Payment of Gratuity Act': ['gratuity calculation', 'section 4', 'forfeiture of gratuity', 'gratuity eligibility', 'employer default', 'gratuity recovery', 'superannuation'],
    'Trade Unions Act': ['trade union registration', 'section 8', 'unfair labor practice', 'recognition dispute', 'union elections', 'membership verification', 'collective bargaining', 'union funds', 'protected workmen', 'union rivalry'],

    # RENT ACT MATTERS
    'Eviction matters of personal necessity': ['bona fide need', 'self-occupation', 'family expansion', 'alternative accommodation', 'comparative hardship', 'landlord requirement', 'residential purpose', 'dependent family', 'business expansion', 'medical necessity'],
    'Eviction matters for re-building': ['structural alteration', 'demolition notice', 'municipal notice', 'reconstruction plan', 'architect certificate', 'building stability', 'redevelopment agreement', 'tenant relocation', 'construction permission', 'completion bond'],
    'Eviction matters of sub-letting': ['unauthorized occupant', 'sub-tenant', 'parting possession', 'license agreement', 'tenancy rights', 'rent receipt', 'tenant verification', 'lock-out period', 'lease violation', 'third party possession'],
    'Arrears of rent': ['rent default', 'monthly rent', 'standard rent', 'rent enhancement', 'deposit scheme', 'tenant default', 'arrears calculation', 'interest on arrears', 'money decree', 'rent receipt'],
    'Enhancement of rent': ['fair rent', 'market rate', 'rent control act', 'cost index', 'capital value', 'improvement cost', 'section 6', 'prevailing rent', 'landlord application', 'tenant objection'],

    # DIRECT TAXES MATTER
    'Income Tax Reference': ['substantial question of law', 'section 256', 'high court reference', 'tax tribunal order', 'case stated', 'appellate reference', 'tax evasion', 'assessment year', 'revision petition', 'tax deduction'],
    'Wealth Tax Act': ['net wealth', 'asset valuation', 'urban land','unproductive assets', 'valuation date', 'wealth statement', 'exempt assets', 'jewellery valuation', 'section 17 wealth tax','wealth tax return'],
    'Capital Gains': ['capital asset', 'transfer u/s 2(47)', 'indexation benefit', 'long term gain', 'cost inflation index', 'section 54', 'exemption claim', 'sale consideration', 'capital gains bond', 'stamp duty value'],
    'Re-assessment': ['reopening assessment', 'section 147', 'income escapement', 'reasons recorded', 'change of opinion', 'fresh material', 'limitation period', 'notice validity', 'concealment income', 'undisclosed investment'],
    'Settlement Commission': ['settlement application', 'full disclosure', 'immunity petition', 'settlement order', 'undisclosed income', 'section 245C', 'commission hearing', 'settlement terms', 'case pending', 'finality of order'],

    # CRIMINAL MATTERS
    'Capital punishment': ['rarest of rare', 'death sentence confirmation', 'section 302 IPC', 'aggravating factors', 'mitigating circumstances', 'deterrent punishment', 'execution warrant', 'mercy petition', 'death row convict', 'commutation plea'],
    'Dowry death': ['section 304B', 'dowry demand', 'soon before death', 'cruelty evidence', 'stridhan recovery', 'domestic violence', 'marriage harassment', 'dowry prohibition', 'bride burning', 'death within 7 years'],
    'Prevention of Corruption Act': ['illegal gratification', 'trap case', 'disproportionate assets', 'section 13', 'public servant', 'vigilance inquiry', 'sanction prosecution', 'bribe money', 'demand proof', 'recovery certificate'],
    'NDPS Act': ['commercial quantity', 'psychotropic substance', 'section 37', 'conscious possession', 'sampling procedure', 'independent witness', 'mandatory minimum', 'contraband seizure', 'drug analysis', 'cartel involvement'],
    'Sexual harassment': ['section 354', 'modesty outrage', 'voyeurism', 'stalking', 'workplace harassment', 'POSH Act', 'zero FIR', 'acid attack', 'compensation award', 'victim protection'],

    # SERVICE MATTERS
    'Promotion': ['seniority cum merit', 'departmental promotion', 'reservation roster', 'sealed cover procedure', 'dpc meeting', 'eligible list', 'promotion criteria', 'benchmarking', 'notional promotion', 'promotion policy'],
    'Pension': ['pension revision', 'commutation amount', 'family pension', 'gratuity payment', 'pension sanction', 'qualifying service', 'invalid pension', 'delay in pension', 'pension cut', 'pensionary benefits'],
    'Disciplinary proceedings': ['charge memo', 'departmental inquiry', 'evidence act violation', 'punishment order', 'minor penalty', 'major penalty', 'cvc guidelines', 'inquiry officer', 'defense assistant', 'ex parte inquiry'],
    'Reservation in service': ['creamy layer reservation','reservation policy service','mandal commission', 'roster system', 'carry forward rule', 'promotion reservation', 'reservation ceiling', 'social backwardness', 'income criteria', 'caste certificate'],
    'Voluntary Retirement': ['vr scheme', 'acceptance condition', 'resignation difference', 'pension eligibility', 'notice period', 'vr benefits', 'forced retirement', 'section 18', 'voluntary separation', 'golden handshake'],

    # INDIRECT TAXES MATTERS
    'Interpretation of the Customs Act': ['customs valuation', 'bill of entry', 'section 14', 'import manifest', 'baggage rules', 'prohibited goods', 'customs duty exemption', 'project imports', 'redeployment certificate', 'customs appeal'],
    'Central Excise Act': ['excisable goods', 'cenvat credit', 'manufacturing process', 'rule 6', 'clearance certificate', 'duty demand notice', 'modvat', 'central excise tariff', 'factory gate', 'removal of goods'],
    'Service Tax': ['taxable service', 'reverse charge', 'mega exemption', 'place of provision', 'point of taxation', 'export of service', 'input service distributor', 'service tax audit', 'abatement claim', 'works contract'],
    'Anti Dumping Duty': ['margin of dumping', 'injury determination', 'designated authority', 'like article', 'normal value', 'landed value', 'provisional duty', 'sunset review', 'domestic industry', 'reference price'],

    # LAND ACQUISITION & REQUISITION
    'Compensation challenges': ['market value', 'section 23', 'solatium', 'additional compensation', 'potential value', 'expert valuation', 'land acquisition award', 'belting system', 'severance compensation', 'injurious affection'],
    'Defence acquisition': ['urgency clause', 'section 17 land acquisition','defence production', 'strategic purpose', 'emergency acquisition', 'national security', 'encroachment removal', 'defence installation', 'buffer zone', 'restricted area'],
    # ACADEMIC MATTERS
    'Examination matters': ['revaluation', 'grace marks', 'malpractice', 'answer sheet inspection', 'moderation policy', 'supplementary exam', 're-test order', 'paper leakage', 'unfair means', 'marking scheme'],
    'Educational management': ['deemed university', 'minority institution', 'mandatory disclosure', 'inspection report''fee regulation', 'affiliation withdrawal', 'teacher qualification', 'student union', 'anti-ragging', 'reservation policy education',],
    # LETTER PETITION & PIL MATTERS
    'Environmental PIL': ['carbon emissions', 'coastal regulation', 'forest clearance', 'wildlife protection', 'environment clearance', 'biosphere reserve', 'polluter pays', 'sustainable development', 'remediation plan', 'environmental audit'],
    'Human rights PIL': ['custodial death', 'prison reform', 'manual scavenging', 'child rights', 'rehabilitation plan', 'bonded labour', 'compensation scheme', 'victim protection', 'legal aid', 'juvenile justice'],

    # ELECTION MATTERS
    'Election petitions': ['corrupt practice', 'booth capturing', 'false affidavit', 'election expenses', 'nomination rejection', 'vote recount', 'election symbol', 'result declaration', 'electoral bonds', 'model code violation'],
    'MP/MLA disqualification': ['office of profit', 'defection law', 'anti-defection', 'resignation validity', 'membership cessation', 'disqualification petition', 'floor test', 'constitutional post', 'legislative privileges', 'breach of oath'],

    # COMPANY LAW, MRTP, SEBI
    'SEBI matters': ['insider trading', 'substantial acquisition', 'takeover code', 'disclosure norms', 'fraudulent trade', 'FII regulations', 'creamy layer investor','delisting shares', 'buyback offer', 'open offer'],
    'Competition Commission': ['anti-competitive', 'abuse of dominance', 'cartelization', 'combination regulation', 'leniency application', 'turnover penalty', 'market share', 'predatory pricing', 'vertical agreement', 'dominant position'],

    # ARBITRATION MATTERS
    'Arbitration challenges': ['arbitral award', 'section 34', 'patent illegality', 'public policy', 'arbitrator appointment', 'jurisdictional error', 'unilateral appointment', 'emergency arbitrator', 'arbitration clause', 'seat vs venue'],

    # COMPENSATION MATTERS
    'Railway accidents': ['untoward incident', 'running train', 'track negligence', 'railway liability', 'passenger ticket', 'level crossing', 'compensation tariff', 'railway tribunal', 'Fatal Accident Act', 'dependency claim'],
    'Telecom disputes': ['call drop', 'spectrum charges', 'interconnection fee', 'quality of service', 'tariff fixation', 'licence fee', 'port charges', 'access deficit', 'subscriber compensation', 'network failure'],

    # HABEAS CORPUS
    'Habeas Corpus': ['illegal detention', 'custody certificate', 'production order', 'missing person', 'wrongful confinement', 'custody jurisdiction', 'habeas corpus writ', 'preventive detention', 'custody transfer', 'detention validity'],

    # APPEAL AGAINST STATUTORY BODIES
    'Tribunal appeals': ['NCLT order', 'SAT decision', 'TDSAT ruling', 'appellate tribunal', 'technical member', 'jurisdictional error', 'perverse finding', 'substantial question', 'limitation period', 'pre-deposit'],

    # FAMILY LAW
    'Child custody': ['best interest', 'guardianship', 'Hague convention', 'parental alienation', 'access rights', 'child abduction', 'welfare principle', 'shared custody', 'visitation rights', 'custody jurisdiction'],
    'Muslim marriage': ['mehr', 'triple talaq', 'iddat period', 'muta marriage', 'nikahnama', 'dower debt', 'maintenance cap', 'khula', 'faskh', 'mahr dispute'],

    # CONTEMPT OF COURT
    'Civil contempt': ['order violation', 'undertaking breach', 'willful disobedience', 'compliance report', 'contempt notice', 'purge contempt', 'apology tendered', 'contempt jurisdiction', 'stay violation', 'court order'],

    # ORDINARY CIVIL MATTERS
    'Specific performance': ['readiness willingness', 'contract enforcement', 'alternative relief', 'section 20', 'discretionary relief', 'mutual consent', 'time essence', 'part performance', 'contract validity', 'title clearance'],
    'Electricity disputes': ['tariff order', 'cross subsidy', 'open access', 'wheeling charges', 'electricity theft', 'connection denial', 'meter tampering', 'regulatory asset', 'power purchase', 'renewable obligation'],

    # BENCH STRENGTH
    'Constitution Bench': ['basic structure doctrine', 'article 145(3) reference','constitutional amendment validity','federalism dispute resolution','judicial independence challenge','land acquisition constitutional validity','reservation constitutional limit','presidential reference article 143','kesavananda bharati ratio','constitutional morality violation'],
    # APPOINTMENTS
    'Judicial appointments': ['collegium system', 'memorandum procedure', 'seniority norm', 'elevation criteria', 'judicial independence', 'appointment delay', 'zone consideration', 'merit vs seniority', 'parent high court', 'additional judge'],

    # PERSONAL LAW
    'Inheritance disputes': ['coparcenary rights', 'ancestral property', 'testamentary succession', 'hindu succession', 'muslim inheritance', 'christian succession', 'parsi succession', 'legal heir', 'succession certificate', 'joint family'],

    # RELIGIOUS ENDOWMENTS
    'Temple management': ['mathadhipati', 'shebait rights', 'idol juristic', 'religious endowment', 'trustee removal', 'dharmakarta', 'archaka appointment', 'temple funds', 'secular management', 'religious practice'],

    # MERCANTILE LAWS
    'Banking disputes': ['sarfaesi', 'npa classification', 'debt recovery', 'wilful defaulter', 'guarantor liability', 'account fraud', 'cheque bounce', 'bank guarantee', 'loan recall', 'priority sector'],

    # JUDICIARY MATTERS
    'Judicial service': ['all india service', 'judicial academy', 'court infrastructure', 'judicial accountability', 'case management', 'judicial independence', 'court automation', 'judicial ethics', 'work allocation', 'judicial discipline'],

    # MEDICAL EDUCATION
    'NEET disputes': ['neet ug', 'all india quota', 'state reservation', 'marks normalization', 'omr challenge', 'percentile system', 'eligibility criteria', 'exam postponement', 'counselling process', 'nri quota'],

    # GOVERNMENT CONTRACTS
    'Tender disputes': ['bid eligibility', 'technical score', 'arbitrary rejection', 'pre-qualification', 'tender condition', 'commercial bid', 'L1 rejection', 'blacklisting order', 'bid security', 'tender cancellation'],

    # MINES & MINERALS
    'Mining leases': ['renewal rejection', 'mining plan', 'environment clearance', 'royalty payment', 'stamp duty', 'mining auction', 'captive mining', 'mineral rights', 'quarry lease', 'district mineral'],

    # CONSUMER PROTECTION
    'Deficiency of service': ['deficiency defined', 'medical negligence', 'housing delay', 'banking service', 'insurance claim', 'telecom service', 'professional service', 'deficiency compensation', 'unfair contract', 'product liability'],

    # ARMED FORCES
    'Military law': ['court martial', 'summary trial', 'command responsibility', 'military pension', 'service record', 'promotion policy', 'resettlement benefits', 'disability pension', 'martial law', 'army act'],

    # CONSTITUTIONAL MATTERS
    'Federal disputes': ['inter-state water', 'border dispute', 'language rights', 'state autonomy', 'governor powers', 'president rule', 'legislative competence', 'concurrent list', 'residuary powers', 'central ordinance']
}

def classify_legal_case(text):
    category_scores = {}

    for category, keywords in category_keywords.items():
        score = sum(text.count(kw) for kw in keywords)
        category_scores[category] = score

    max_score = max(category_scores.values())
    if max_score == 0:
        return 'Others'

    top_categories = [cat for cat, score in category_scores.items() if score == max_score]

    return top_categories[0] if len(top_categories) == 1 else ', '.join(top_categories)

ik['Category'] = ik['cleaned_text'].apply(classify_legal_case)
