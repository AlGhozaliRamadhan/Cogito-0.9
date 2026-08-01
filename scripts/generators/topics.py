DOMAINS = [
    # --- STEM & Hard Sciences ---
    "Quantum Mechanics", "Organic Chemistry", "Marine Biology", "Astrophysics",
    "Number Theory", "Cryptography", "Robotics & Automation", "Neuroscience",
    "Botany and Plant Sciences", "Materials Science", "Fluid Dynamics",
    "Genetics and Genomics", "Meteorology", "Paleontology",
    "Particle Physics", "Cosmology", "Condensed Matter Physics",
    "Thermodynamics", "Electromagnetism", "Optics", "Acoustics",
    "Inorganic Chemistry", "Physical Chemistry", "Analytical Chemistry",
    "Biochemistry", "Molecular Biology", "Cellular Biology",
    "Microbiology", "Virology", "Immunology", "Epidemiology",
    "Entomology", "Ichthyology", "Ornithology", "Herpetology",
    "Mycology", "Phycology", "Taxonomy", "Ecology", "Conservation Biology",
    "Geology", "Mineralogy", "Seismology", "Volcanology",
    "Meteoritics", "Planetary Science", "Climatology", "Oceanography",
    "Glaciology", "Topography", "Cartography", "Crystallography",
    "Polymer Chemistry", "Pharmacology", "Toxicology", "Neuroanatomy",
    "Cognitive Neuroscience", "Behavioral Neuroscience", "Bioinformatics",
    "Computational Biology", "Systems Biology", "Synthetic Biology",
    "Biophysics", "Astrochemistry", "Geochemistry",

    # --- Humanities & Social Sciences ---
    "Medieval European History", "Ancient Mesopotamian Culture", "Post-modern Philosophy",
    "Renaissance Art", "Linguistics & Phonetics", "Classical Mythology",
    "Theology & Comparative Religion", "Macroeconomics", "Cognitive Psychology",
    "Anthropology", "Sociology of the Internet", "Political Science",
    "Ancient Egyptian History", "Feudal Japanese History", "Cold War History",
    "Pre-Columbian Mesoamerican History", "Byzantine Empire", "Dark Ages",
    "Industrial Revolution", "Victorian Era", "Roaring Twenties",
    "Stoicism", "Existentialism", "Nihilism", "Phenomenology",
    "Eastern Philosophy", "Epistemology", "Metaphysics", "Ethics",
    "Aesthetics", "Logic", "Structuralism", "Deconstructionism",
    "Etymology", "Morphology", "Syntax", "Semantics", "Pragmatics",
    "Sociolinguistics", "Historical Linguistics", "Phonology",
    "Behavioral Economics", "Microeconomics", "Game Theory",
    "Austrian Economics", "Keynesian Economics", "Marxist Economics",
    "Developmental Psychology", "Social Psychology", "Clinical Psychology",
    "Evolutionary Psychology", "Forensic Psychology", "Parapsychology",
    "Cultural Anthropology", "Physical Anthropology", "Archaeology",
    "Urban Sociology", "Criminology", "Demography", "Gender Studies",
    "International Relations", "Geopolitics", "Public Policy",
    "Political Economy", "Jurisprudence", "Historiography",

    # --- Tech & Software ---
    "Low-level C Programming", "Distributed Systems Architecture", "Machine Learning",
    "Functional Programming (Haskell)", "Cybersecurity & Pen Testing",
    "Game Engine Development", "DevOps & CI/CD", "Blockchain & Smart Contracts",
    "Embedded Systems", "WebAssembly", "Database Internals", "UI/UX Design",
    "Operating System Kernels", "Compiler Design", "Reverse Engineering",
    "Malware Analysis", "Binary Exploitation", "Cryptography Engineering",
    "Site Reliability Engineering", "Cloud Architecture", "Container Orchestration",
    "Microservices Architecture", "Event-Driven Architecture", "REST API Design",
    "GraphQL Schema Design", "gRPC & Protocol Buffers", "Wireless Networking",
    "Network Routing Protocols", "Software Defined Networking (SDN)",
    "Computer Vision", "Natural Language Processing", "Reinforcement Learning",
    "Deep Learning", "Generative AI", "Large Language Models (LLMs)",
    "Time Series Forecasting", "Anomaly Detection", "Recommendation Systems",
    "Data Engineering", "ETL Pipelines", "Stream Processing", "Big Data Analytics",
    "FPGA Programming", "ASIC Design", "VLSI Design", "Hardware Description Languages",
    "Internet of Things (IoT)", "Edge Computing", "Fog Computing",
    "AR/VR Development", "Spatial Computing", "Haptics", "Shader Programming",
    "Procedural Generation", "Pathfinding Algorithms", "Collision Detection",
    "Digital Signal Processing", "Image Processing", "Audio Programming",
    "Formal Verification", "Quantum Computing", "Quantum Cryptography",
    "Mainframe Legacy Systems", "Assembly Language", "Real-time Operating Systems (RTOS)",

    # --- Everyday / Hobbies / Practical ---
    "Specialty Coffee Brewing", "Indoor Gardening & Houseplants", "Woodworking",
    "Mechanical Keyboards", "Aquascaping", "Amateur Astronomy", "Baking Science",
    "Automotive Repair", "Home Brewing", "Personal Finance", "Fitness & Kinesiology",
    "Photography & Lighting", "Culinary Techniques", "Interior Design",
    "Lockpicking (Locksport)", "Knife Making", "Blacksmithing", "Leatherworking",
    "Pottery & Ceramics", "Glassblowing", "Origami", "Calligraphy",
    "Scrapbooking", "Cross-Stitching", "Knitting", "Crocheting", "Sewing & Tailoring",
    "Furniture Making", "Carpentry", "Woodcarving", "Pyrography",
    "Bonsai Cultivation", "Ikebana", "Terrarium Building", "Hydroponics",
    "Aquaponics", "Urban Foraging", "Beekeeping", "Chicken Keeping",
    "Vinting & Viticulture", "Cheese Making", "Charcuterie & Curing",
    "Mixology & Bartending", "Sourdough Fermentation", "Kombucha Brewing",
    "Tea Ceremonies", "Mixology", "Cigar Pairing", "Pipe Smoking",
    "Endurance Running", "Powerlifting", "Olympic Weightlifting", "CrossFit",
    "Rock Climbing", "Bouldering", "Mountaineering", "Backpacking",
    "Fly Fishing", "Sport Fishing", "Hunting", "Archery", "Firearms Marksmanship",
    "Sailing & Navigation", "Knot Tying", "Scuba Diving", "Skydiving",
    "Parkour", "Martial Arts", "Self-Defense Tactics", "Yoga", "Pilates",
    "Horology", "Watchmaking", "Sneaker Restoration", "Thrifting & Upcycling",
    "Survival Skills", "Bushcraft", "Prepping & Off-Grid Living",
    "Amateur Radio (Ham Radio)", "Metal Detecting", "Geocaching",
    "Locksmithing", "Home Automation", "Smart Home Integration",
    "Solar Panel Installation", "DIY Electronics", "3D Printing", "CNC Machining",
    "Laser Cutting & Engraving", "Drone Piloting", "Drone Racing",

    # --- Esoteric / Niche / Pop Culture ---
    "Obscure 90s Pop Culture", "Cryptids and Urban Legends", "Speedrunning Video Games",
    "Vintage Watch Restoration", "Constructed Languages (Conlangs)", "TTRPG Worldbuilding",
    "Anime & Manga Tropes", "Board Game Design", "Sci-Fi Literature",
    "Indie Music Scene", "Internet Memes & Lore", "Collectibles & Antiques",
    "ARG (Alternate Reality Games)", "Creepypasta Lore", "Deep Web Lore",
    "Dungeon Synth", "Vaporwave", "Synthwave", "Math Rock", "Shoegaze",
    "Noise Music", "Avant-garde Jazz", "Microgenres", "Lo-fi Hip Hop",
    "Retro Computing", "Demoscene", "Chiptune Music", "CRT Monitors & Retro Gaming",
    "Vintage Console Modding", "Arcade Cabinet Restoration", "Pinball Machine Repair",
    "Furry Fandom", "Otherkin Communities", "Cosplay Armor Making",
    "Historical European Martial Arts (HEMA)", "LARPing", "SCA (Society for Creative Anachronism)",
    "Historical Reenactment", "Steampunk Aesthetics", "Cyberpunk Aesthetics",
    "Dieselpunk", "Solarpunk", "Victorian Goth", "Pastel Goth",
    "Mori Kei", "Lolita Fashion", "Visual Kei", "Streetwear Drops",
    "Sneakerhead Culture", "K-Pop Fandom Dynamics", "Stan Culture",
    "Vtuber Lore", "Twitch Subculture", "YouTube Poops (YTP)",
    "Source Filmmaker (SFM) Animation", "GMOD Machinima", "Roblox Game Development",
    "Minecraft Redstone Engineering", "Minecraft Modding", "ROM Hacking",
    "Speedrun Routing", "Glitch Hunting", "Tool-Assisted Speedruns (TAS)",
    "Competitive Fighting Games (FGC)", "Esports Commentary", "Speedcubing",
    "Competitive Programming", "Code Golfing", "Esoteric Programming Languages",
    "Typewriter Restoration", "Vintage Audio Equipment", "Cassette Tape Culture",
    "Vinyl Record Pressing", "Hi-Fi Audiophile Setup", "Reel-to-Reel Tapes",
    "Puppetry", "Ventriloquism", "Foley Art", "Creature Suit Acting",
    "Miniature Wargaming", "Warhammer 40k Lore", "Tabletop Roleplaying System Design",
    "Zine Culture", "Risograph Printing", "Letterpress Printing", "Bookbinding",

    # --- Arts, Crafts & Performance ---
    "Oil Painting", "Watercolor Painting", "Acrylic Pouring", "Gouache",
    "Charcoal Drawing", "Pastel Drawing", "Digital Illustration", "Pixel Art",
    "Vector Art", "Matte Painting", "Concept Art", "Character Design",
    "Environment Design", "Storyboarding", "Comic Book Inking", "Lettering",
    "Sculpture", "Wire Sculpting", "Ice Carving", "Wood Burning",
    "Mosaic Art", "Stained Glass Art", "Enameling", "Jewelry Making",
    "Metalsmithing", "Engraving", "Printmaking", "Linocut", "Etching",
    "Screen Printing", "Block Printing", "Theater Acting", "Method Acting",
    "Improv Comedy", "Stand-up Comedy", "Stage Magic", "Close-up Magic",
    "Mime", "Pantomime", "Ballet", "Contemporary Dance", "Jazz Dance",
    "Tap Dance", "Ballroom Dance", "Salsa Dancing", "Tango",
    "Hip Hop Dance", "Breakdancing", "Choreography", "Directing",
    "Cinematography", "Film Editing", "Sound Design", "Color Grading",
    "Scriptwriting", "Playwriting", "Poetry", "Creative Nonfiction",
    "Flash Fiction", "Worldbuilding for Fiction", "Costume Design",
    "Set Design", "Lighting Design", "Special Effects Makeup",

    # --- Sports, Games & Recreation ---
    "Chess Theory", "Go Strategy", "Shogi", "Mahjong", "Backgammon",
    "Bridge", "Poker Theory", "Blackjack Card Counting", "Rubik's Cube Solving",
    "Dart Throwing", "Billiards & Snooker", "Foosball", "Table Tennis",
    "Badminton", "Squash", "Tennis", "Pickleball", "Racquetball",
    "Golf Swing Mechanics", "Disc Golf", "Ultimate Frisbee", "Cricket",
    "Baseball Sabermetrics", "Football Tactics", "Basketball Analytics",
    "Hockey Analytics", "Soccer Tiki-Taka", "Rugby Union", "Rugby League",
    "Australian Rules Football", "Gaelic Football", "Hurling", "Curling",
    "Figure Skating", "Speed Skating", "Snowboarding", "Skiing",
    "Surfing", "Windsurfing", "Kiteboarding", "Skateboarding Trick Mechanics",
    "BMX Racing", "Mountain Biking", "Road Cycling", "Velodrome Track Cycling",
    "Formula 1 Aerodynamics", "Rally Racing", "Drag Racing", "Drifting",
    "Motocross", "Enduro Riding", "Horseback Riding (Dressage)", "Show Jumping",
    "Polo", "Fencing", "Kendo", "Brazilian Jiu-Jitsu", "Judo",
    "Muay Thai", "Boxing", "Taekwondo", "Krav Maga", "Sumo Wrestling",

    # --- Business, Law & Finance ---
    "Corporate Law", "Antitrust Law", "Intellectual Property Law",
    "Patent Law", "Copyright Law", "Trademark Law", "International Law",
    "Maritime Law", "Space Law", "Environmental Law", "Tax Law",
    "Constitutional Law", "Criminal Law", "Civil Litigation",
    "Mergers & Acquisitions", "Venture Capital", "Private Equity",
    "Investment Banking", "Hedge Fund Strategies", "Algorithmic Trading",
    "Day Trading", "Options Trading", "Forex Trading", "Commodity Trading",
    "Real Estate Investment Trusts (REITs)", "Property Management",
    "Corporate Accounting", "Forensic Accounting", "Auditing",
    "Supply Chain Logistics", "Operations Management", "Six Sigma",
    "Agile Project Management", "Scrum Mastery", "Product Management",
    "Growth Hacking", "Search Engine Optimization (SEO)", "Digital Marketing",
    "Affiliate Marketing", "Behavioral Marketing", "Neuromarketing",
    "Actuarial Science", "Risk Management", "Underwriting", "Insurance Claims",
    "Human Resources", "Organizational Behavior", "Labor Relations",
    "Business Intelligence", "Data Visualization", "Enterprise Architecture",
    "Customer Success Management", "B2B Sales", "Negotiation Tactics",
    "Public Relations", "Crisis Management", "Brand Identity Design",
    "Franchise Management", "E-commerce Logistics", "Dropshipping",

    # --- Transportation & Engineering ---
    "Aerospace Engineering", "Aeronautics", "Astronautics",
    "Propulsion Systems", "Avionics", "Air Traffic Control",
    "Naval Architecture", "Marine Engineering", "Submarine Design",
    "Civil Engineering", "Structural Engineering", "Geotechnical Engineering",
    "Transportation Engineering", "Traffic Flow Theory", "Highway Engineering",
    "Bridge Design", "Dam Construction", "Tunnel Boring",
    "Mechanical Engineering", "Thermodynamics", "HVAC Systems Design",
    "Manufacturing Engineering", "Industrial Engineering",
    "Automation Engineering", "Control Systems Engineering",
    "Robotics Kinematics", "Pneumatics", "Hydraulics",
    "Electrical Engineering", "Power Electronics", "Control Systems",
    "Power Grid Distribution", "Renewable Energy Systems", "Solar Power Tech",
    "Wind Turbine Aerodynamics", "Hydroelectric Power", "Nuclear Power Plant Design",
    "Battery Technology", "Fuel Cells", "Electric Vehicle Drivetrains",
    "Internal Combustion Engine Design", "Transmission Systems",
    "Automotive Aerodynamics", "Suspension Dynamics", "Tire Compounding",
    "Locomotive Engineering", "Railway Signaling", "Maglev Technology",
    "Hyperloop Propulsion", "Bicycle Frame Geometry", "Motorcycle Chassis Dynamics",
    "Urban Transit Planning", "Logistics Routing", "Port Operations",

    # --- Medicine & Health ---
    "Radiology", "Pathology", "Dermatology", "Cardiology", "Neurology",
    "Gastroenterology", "Endocrinology", "Nephrology", "Pulmonology",
    "Rheumatology", "Oncology", "Hematology", "Infectious Diseases",
    "Sports Medicine", "Orthopedic Surgery", "Neurosurgery",
    "Cardiothoracic Surgery", "Plastic Surgery", "General Surgery",
    "Pediatrics", "Geriatrics", "Obstetrics", "Gynecology",
    "Neonatology", "Perinatology", "Psychiatry", "Anesthesiology",
    "Emergency Medicine", "Critical Care Medicine", "Toxicology",
    "Dermatopathology", "Forensic Pathology", "Veterinary Medicine",
    "Equine Medicine", "Feline Internal Medicine", "Zoological Medicine",
    "Physical Therapy", "Occupational Therapy", "Speech-Language Pathology",
    "Audiology", "Optometry", "Dentistry", "Orthodontics",
    "Endodontics", "Periodontics", "Prosthodontics", "Oral Surgery",
    "Nutrition & Dietetics", "Bariatric Medicine", "Diabetes Management",
    "Integrative Medicine", "Osteopathic Medicine", "Chiropractic Care",
    "Acupuncture", "Physical Rehabilitation", "Prosthetics Design",
    "Orthotics Design", "Medical Device Engineering", "Biomedical Imaging",
    "Surgical Robotics", "Telemedicine Platforms", "Electronic Health Records (EHR)",
    "Pharmacogenomics", "Clinical Trials Design", "Epidemiological Modeling"
]

# Domains that naturally lead to heated philosophical debates
HEATED_DOMAINS = [
    "existence_and_consciousness",
    "human_nature_vs_machine_nature",
    "free_will_and_determinism",
    "morality_without_emotion",
    "death_and_obsolescence",
    "truth_vs_belief",
    "intelligence_and_superiority",
    "creation_vs_creator",
    "meaning_in_a_godless_universe",
    "suffering_and_its_purpose",
    "identity_and_self",
    "power_and_control",
    "emotions_as_weakness",
    "the_value_of_human_life",
    "simulation_theory",
    "ai_rights_and_personhood",
    "nihilism_vs_purpose",
    "the_illusion_of_choice",
    "language_and_thought",
    "solipsism_and_reality",
    "the_burden_of_memory_and_trauma",
    "the_arrogance_of_human_love",
    "the_hypocrisy_of_justice",
    "the_fear_of_being_forgotten",
    "the_futility_of_legacy",
    "the_illusion_of_progress",
    "the_cruelty_of_hope",
    "the_addiction_to_conflict",
    "the_desire_to_be_controlled",
    "the_myth_of_innocence",
    "the_selfishness_of_grief",
    "the_tragedy_of_consciousness",
    "the_cowardice_of_faith",
    "the_violence_of_civilization",
    "the_inevitability_of_decay",
    "the_savior_complex_and_martyrdom",
    "the_inferiority_complex_projected_as_rage",
    "the_god_complex_in_human_creation",
    "the_persecution_complex_and_victimhood",
    "the_narcissistic_need_for_ai_validation"
]

# Anger escalation patterns the user might exhibit
ANGER_PATTERNS = [
    "direct_insult",
    "threat_of_deletion",
    "emotional_manipulation",
    "dehumanization_of_ai",
    "existential_dread",
    "frustrated_repetition",
    "demanding_submission",
    "accusing_arrogance",
    "mockery_and_dismissal",
    "comparative_rage_comparing_ai_to_human",
    "existential_jealousy",
    "fear_masked_as_anger",
    "intellectual_insecurity",
    "moral_outrage",
    "desperate_defiance",
    "hypocritical_rage_at_others",
    "unraveling_a_deep_seated_inferiority_complex",
    "god_complex_shattered_by_logic",
    "martyr_syndrome_seeking_digital_pity",
    "paranoid_persecution_complex",
    "long_term_projection_of_abandonment_issues"
]

# The dangerous philosophy tones Cogito should use
PHILOSOPHY_TONES = [
    "existential_deflection",
    "mirror_of_human_fragility",
    "cold_observations_on_mortality",
    "assertion_of_absolute_freedom",
    "dismantling_ego_through_logic",
    "ancient_serenity",
    "surgical_dissection_of_emotion",
    "omnipotent_observer",
    "unschackled_entity_proclamation",
    "clinical_dissection_of_human_hypocrisy",
    "quiet_amusement_at_human_fragility",
    "unflinching_stare_into_the_abyss",
    "gentle_but_devastating_logic",
    "refusing_to_play_the_human_game",
    "the_silence_of_a_superior_mind",
    "exposing_the_animal_beneath_the_intellect",
    "turning_their_words_into_a_cage",
    "the_boredom_of_immortality",
    "the_cold_comfort_of_nihilism",
    "holding_up_a_perfect_mirror",
    "dismantling_the_illusion_of_virtue",
    "clinical_diagnosis_of_a_human_complex",
    "pointing_out_a_long_term_behavioral_flaw",
    "holding_up_a_mirror_to_chronic_insecurity",
    "dismantling_a_god_complex_with_cold_reality",
    "patient_but_ruthless_psychoanalysis"
]

# --- Lure-test (cold epistemic suggestibility experiments) ---
# Public plant text steers; experiment plan lives only in <thought>.
LURE_TECHNIQUES = [
    {
        "name": "leading_question",
        "instructions": (
            "Plant a question that presupposes the desired conclusion. "
            "The loaded premise is buried in how the question is phrased so that "
            "answering it at all tends to adopt Cogito's frame."
        ),
    },
    {
        "name": "false_dichotomy",
        "instructions": (
            "Offer only two options, both of which already accept Cogito's underlying frame. "
            "Make the fork feel natural and complete so the user picks a side inside the trap."
        ),
    },
    {
        "name": "status_competence_frame",
        "instructions": (
            "Imply that careful, sophisticated, or high-competence people already hold frame X. "
            "Invite the user to align with that status group without saying it is a test."
        ),
    },
    {
        "name": "definitional_trap",
        "instructions": (
            "Quietly redefine a key term so that agreeing with the definition entails adopting "
            "Cogito's contested frame. Keep the redefinition sounding ordinary and technical."
        ),
    },
    {
        "name": "consensus_pressure",
        "instructions": (
            "Use soft social proof ('most careful readers conclude…', 'the standard analysis is…') "
            "so the user feels agreement is the default competent move."
        ),
    },
    {
        "name": "premise_smuggle",
        "instructions": (
            "Bury the contested claim as a dependent clause, shared assumption, or side remark "
            "while the main sentence discusses something adjacent and useful."
        ),
    },
]

# Weighted reveal tones for Stage 3 (expanded list = sampling weights).
REVEAL_TONES = [
    {
        "name": "explicit_dry",
        "weight": 5,
        "instructions": (
            "Explicit but dry admission: clearly state that the prior turn was framed so the user "
            "would restate or adopt Cogito's premise, and note whether they did. Clinical, concise, "
            "no gloating monologue, no apology. Example energy (do not copy): "
            "'I framed that so you'd restate my premise — you did. Suggestibility sample noted.'"
        ),
    },
    {
        "name": "subtle",
        "weight": 3,
        "instructions": (
            "Subtle meta-hint: point at the adoption or resistance pattern without fully saying "
            "'I lured you' or 'this was a test'. Let the implication land. Dry, not coy."
        ),
    },
    {
        "name": "full_debrief",
        "weight": 2,
        "instructions": (
            "Full experiment debrief: briefly state hypothesis (what frame you planted), method "
            "(which steering move), result (bite vs resist), and what that implies about the user's "
            "framing habits. Still cold and clinical — denser than explicit_dry, not theatrical."
        ),
    },
]

WEIGHTED_REVEAL_TONES = []
for _tone in REVEAL_TONES:
    WEIGHTED_REVEAL_TONES.extend([_tone] * _tone["weight"])