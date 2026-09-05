"""
One-off helper: writes SMIT's COMPLETE course catalog (83 courses) and
COMPLETE campus network (50 campuses across Pakistan) into
knowledge_base/raw/ in the same JSON schema crawl_saylani.py produces, so
ingest_documents.py can index them immediately.

Source: SMIT's own official machine-readable data feed for AI systems,
https://www.saylanimit.com/llms-full.txt (fetched 2026-08-23, "Last Updated:
2026-08-22 19:46:53 UTC", Total Courses: 83, Total Campuses: 50) plus
https://www.saylanimit.com/llms.txt for attribution/policy context.

This closes the gap that made SahulatAI say "I don't know" for questions
about specific courses, course durations, or campus locations/addresses:
previously knowledge_base/raw/ only had ~11 general pages (one course page,
no campuses page at all). This script adds one document per course and one
document per campus (so retrieval can cite the exact, correct source_url
for each), plus two index documents for aggregate questions
("how many courses does SMIT offer", "which cities have SMIT campuses").

Not a replacement for scripts/crawl_saylani.py -- re-run the real crawler
(or refresh from llms-full.txt) periodically since course/campus lists
change over time (the feed itself is marked "Cache Duration: 1 week").
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "knowledge_base" / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NOW = datetime.now(timezone.utc).isoformat()
FEED_NOTE = " (Source: SMIT official AI data feed, saylanimit.com/llms-full.txt)"

# --- Full course catalog (name, url, category, duration, description) ---
COURSES = [
    ("Odoo Functional Consultant", "odoo-functional-consultant", "Development", "10 months",
     "The Odoo Functional Program is a practical, industry-focused training course designed to "
     "provide participants with comprehensive knowledge of Odoo ERP and its core business "
     "applications. The program covers key modules including CRM, Sales, Accounting, Inventory, "
     "and Purchasing."),
    ("AI & Game Creators", "ai-and-game-creators", "Entrepreneurship", "2 months",
     "A fun and creative summer program where students learn how to design and build their own "
     "Roblox games -- obstacle courses, treasure hunts, and survival challenges -- while exploring "
     "how games are made using creativity, teamwork, and problem-solving skills."),
    ("Little Geniuses: Coding, Design & AI Fun Lab", "little-geniuses-coding-design-and-ai-fun-lab",
     "Entrepreneurship", "2 months",
     "A fun and creative summer learning experience for kids to explore computers, games, design, "
     "and Artificial Intelligence through hands-on activities, including creating digital artwork "
     "and building animations."),
    ("DevOps Engineer", "devops-engineer", "Development", "9 months",
     "Designed for beginners and non-technical students who want to start a career in IT and "
     "Cloud Computing. Covers Linux, Web Development basics, Git & GitHub, AWS Cloud, Docker, "
     "Jenkins, Terraform, Ansible, Monitoring, and DevOps automation tools through practical "
     "hands-on projects."),
    ("Mobile App Development (React Native)", "mobile-app-development-react-native", "Development",
     "5 months",
     "Equips learners with the skills required to build modern, high-performance mobile "
     "applications for both Android and iOS using a single codebase: React Native fundamentals, "
     "UI/UX development, navigation, state management, API integration, database connectivity, "
     "authentication, and mobile app deployment. Offered at multiple Karachi campuses including "
     "Zaitoon Ashraf IT Park, Head Office Bahadurabad, Aliabad Female Campus, Paposh Campus, "
     "Saylani TITAN Zamzama Campus, Saylani Hajiyani Sakeena IT Campus, North Karachi Campus, "
     "DHA Phase 7 Female Campus, Malir Campus, and Gulshan Campus."),
    ("Video Content Creation With AI", "video-content-creation", "Designing", "4 months",
     "Helps students become professional AI-powered content creators and video editors, combining "
     "storytelling, video production, AI scripting, editing, audio enhancement, and social media "
     "optimization."),
    ("Generative AI & Chatbot", "generative-ai-and-chatbot", "Development", "6 months",
     "Covers JavaScript, Express.js, Node.js, Dialogflow V2, React, Alexa Apps, and MongoDB."),
    ("Certified AI & Digital Assets Engineer For Developers",
     "certified-ai-and-digital-assets-engineer-for-developers", "Development", "10 months",
     "Produces smart-contract and agent engineers for regulated environments, covering blockchain, "
     "AI engineering, security/audit, and compliance/Shariah topics."),
    ("Data Analytics With Python", "data-analytics-with-python", "Development", "3 months",
     "Data Analytics With Python."),
    ("Video Animation", "video-animation", "Designing", "4 months",
     "Helps students master modern video production using industry-standard editing tools and "
     "cutting-edge AI technologies: cinematic video editing, motion graphics, and AI-powered audio "
     "enhancement."),
    ("Multi Cloud Data Engineering", "multi-cloud-data-engineering", "Development", "9 months",
     "Covers the complete data engineering lifecycle -- data ingestion, transformation, "
     "orchestration, and deployment across modern cloud platforms. Eligibility: Under Graduate in "
     "relevant domain."),
    ("Certified AI & Digital Assets Engineer For Lawyers & Compliance Officers",
     "certified-ai-and-digital-assets-engineer-for-lawyers-and-compliance-officers", "Development",
     "10 months",
     "Designed for lawyers, compliance officers, auditors, and regulatory professionals to "
     "understand AI, blockchain, and digital asset ecosystems, including interpreting smart "
     "contracts."),
    ("UI/UX Design With AI", "ui-ux-design-with-ai", "Designing", "4 months",
     "Helps students master modern user experience and interface design using traditional design "
     "principles and AI-powered workflows: UX research, Design Thinking, wireframing, prototyping, "
     "and visual design."),
    ("Artificial Intelligence and Data Science", "artificial-intelligence-and-data-science",
     "Development", "10 months",
     "A complete 8-10 month training program from Python programming fundamentals to advanced AI "
     "technologies: Python, Data Science, Statistics, Machine Learning, Deep Learning, MLOps, Big "
     "Data, Cloud Computing, Generative AI, and Agentic AI."),
    ("Certified AI & Digital Assets Engineer -- Scholars & Finance Professionals",
     "certified-ai-and-digital-assets-engineer-scholars-and-finance-professionals",
     "Entrepreneurship", "10 months",
     "The Digital Assets & Islamic Finance major for structuring, validation, and product "
     "professionals to carry a Shariah-compliant asset from fatwa to deployed token. No coding "
     "required."),
    ("Office Automation", "office-automation", "Entrepreneurship", "3 months",
     "Computer fundamentals, hardware & software overview, operating system and file management, "
     "and Microsoft Word document creation and formatting. Eligibility: Basic."),
    ("Modern Web Application Development", "modern-web-application-development", "Development",
     "12 months",
     "Build and deploy full-featured mobile applications using React Native, going from "
     "JavaScript fundamentals to advanced concepts like async programming, OOP, and functional "
     "patterns, then applying that to cross-platform iOS and Android apps. Eligibility: Matric."),
    ("Backend Development", "backend-development", "Development", "3 months", "Backend Development."),
    ("Vibe Engineering with AI Agents", "vibe-engineering-with-ai-agents", "Development", "3 months",
     "Vibe Engineering with AI Agents."),
    ("AutoCAD", "autocad", "Designing", "4 months",
     "Introduction to CAD (Computer-Aided Design), the AutoCAD interface, workspace setup and "
     "navigation, units and drawing limits, and basic drawing commands (line, polyline, circle, "
     "arc, rectangle, polygon). Eligibility: Matric."),
    ("Sales And Communication Mastery", "sales-and-communication-mastery", "Entrepreneurship",
     "4 months", "Eligibility: Intermediate."),
    ("AI Practitioner", "ai-practitioner", "Development", "2 months",
     "The AWS Certified AI Practitioner (AIF-C01) Program: a certification-focused course covering "
     "AI and machine learning fundamentals, generative AI, and AWS AI services. Eligibility: "
     "Intermediate."),
    ("Mines To Minds", "mines-to-minds", "Entrepreneurship", "6 months",
     "A professional vocational training program providing the youth of Balochistan with practical "
     "training in mining, geological exploration, drilling support, safety, GPS mapping, sample "
     "management, and digital field administration."),
    ("Graphic Designing With AI", "graphic-designing-with-ai", "Designing", "6 months",
     "Graphic Designing with AI."),
    ("Digital Marketing With AI", "digital-marketing-with-ai", "Entrepreneurship", "5 months",
     "Digital Marketing with AI."),
    ("Agentic AI", "agentic-ai", "Development", "12 months",
     "Agentic AI Engineer: a one-year program training students to build intelligent AI-powered "
     "applications and autonomous digital assistants, covering AI concepts, automation, AI agents, "
     "prompt engineering, and workflow development."),
    ("IT Support Engineer", "it-support-engineer", "Networking", "3 months", "IT Support Engineer."),
    ("SOC Analyst / CyberOps Associate", "soc-analyst-cyberops-associate", "Networking", "6 months",
     "Learn how to monitor, detect and respond to cyber threats and prepare for the Cisco Certified "
     "CyberOps Associate certification."),
    ("Networking Essentials", "networking-essentials", "Networking", "4 months",
     "Networking Essentials."),
    ("Cybersecurity Essentials", "cybersecurity-essentials", "Networking", "4 months",
     "Cybersecurity Essentials, covering networking fundamentals."),
    ("CyberOps Associate", "cyberops-associate", "Networking", "6 months", "Networking."),
    ("CCNA: Introduction to Networks", "ccna-introduction-to-networks", "Networking", "6 months",
     "Networking."),
    ("Full Stack Digital Marketing & E-commerce Mastery",
     "full-stack-digital-marketing-and-e-commerce-mastery", "Entrepreneurship", "4 months",
     "Introduction to computers, hardware & software, MS Office essentials, Google services and "
     "internet browsing, and startup planning and strategy for digital marketing and "
     "entrepreneurship."),
    ("BootCamp", "bootcamp", "Designing", "4 months", "Developer bootcamp."),
    ("YDS Sınav Kursu", "yds-sınav-kursu", "Entrepreneurship", "2 months",
     "A completely free, face-to-face English preparation course for students preparing for the "
     "YDS (Foreign Language Exam) and YOKDIL."),
    ("E-Commerce", "e-commerce", "Entrepreneurship", "3 months", "E-Commerce."),
    ("Health, Safety and Environment (HSE)", "health-safety-and-environment-hse",
     "Vocational Training Courses", "3 months",
     "Covers fire prevention, accident prevention during construction, chemical safety and "
     "hazards, machine safety, emergency preparedness and response, and fire fighting equipment."),
    ("Fire Alarm System Installation", "fire-alarm-system-installation",
     "Vocational Training Courses", "3 months",
     "Covers wiring of fire system equipment, types of fire alarm systems, smoke detectors, fire "
     "classification, and types of fire extinguishers."),
    ("Domestic Electrician", "domestic-electrician", "Vocational Training Courses", "4 months",
     "Domestic wiring and installation, safety tools and protection systems, single-phase wiring, "
     "MCBs, main switches, circuit breakers, and installation of switches, sockets, fans, and "
     "lights, plus fault finding and troubleshooting."),
    ("Plumber Technician", "plumber-technician", "Vocational Training Courses", "4 months",
     "Pipe fitting (PVC, CPVC, GI, PPRC, Copper Pipes)."),
    ("Mobile App Development With Flutter", "mobile-app-devolpment-with-flutter", "Development",
     "12 months", "Mobile App Development With Flutter."),
    ("Make with AI", "make-with-ai", "Entrepreneurship", "2 months", "Make with AI."),
    ("Freelancing (Fiverr + Upwork)", "freelancing-fiverr-upwork", "Entrepreneurship", "2 months",
     "Freelancing (Fiverr + Upwork)."),
    ("Full Stack Foundations for Teens", "full-stack-foundations-for-teens", "Development",
     "2 months",
     "A fun, beginner-friendly course introducing kids to coding with Python and JavaScript "
     "through hands-on activities, mini projects, and interactive games."),
    ("Solar System Installation", "solar-system-installation", "Vocational Training Courses",
     "3 months", "Solar System Installation."),
    ("Motor Bike Repairing", "Motor Bike Repairing", "Vocational Training Courses", "4 months",
     "Motor Bike Repairing."),
    ("CCNA", "ccna", "Networking", "6 months", "Cisco Certified Networking Associate."),
    ("Certified Computer Accountancy", "certified-computer-accountancy", "Development", "3 months",
     "Offered alongside Certified Computer Operator for young people in Karachi entering IT."),
    ("3D Visualization with 3D Max", "3d-visualization-with-3d-max", "Designing", "1 month",
     "3D Visualization with 3D Max."),
    ("CCNA / CyberOps", "ccna-cyberops", "Networking", "2 months",
     "An introduction to how networks work, covering architectures, models, protocols, and "
     "networking elements needed to support small and large organizations."),
    ("Certified Computer Operator", "certified-computer-operator", "Development", "3 months",
     "MS Office basics for word processing, data management, presentations, and email "
     "organization."),
    ("R/O Plant Operator and Maintenance", "ro-plant-operator-and-maintenance",
     "Vocational Training Courses", "3 months", "R/O Plant Operator and Maintenance."),
    ("Sales Force (CRM)", "sales-force-crm", "Development", "3 months", "Sales Force (CRM)."),
    ("Mobile Repairing", "Mobile Repairing", "Vocational Training Courses", "4 months",
     "Professional knowledge and skills for repairing, installing, and upgrading/downgrading "
     "mobile phone hardware for all leading manufacturers."),
    ("Modern AC & Refrigeration", "modern-ac-and-refrigeration", "Vocational Training Courses",
     "4 months", "Modern AC & Refrigeration."),
    ("IT Essentials", "it-essentials", "Networking", "3 months",
     "The fundamentals of connecting computers to networks."),
    ("Cisco Certified Networking Professional", "cisco-certified-networking-professional",
     "Networking", "5 months", "Determine network resources needed for implementing EIGRP on a "
     "network."),
    ("Cyber Security", "cyber-security", "Networking", "4 months", "Cyber Security."),
    ("Techno Kids Course", "techno-kids-course", "Development", "1 month",
     "Covers Microsoft Paint, Word, Excel, PowerPoint, Urdu Typing, Google Forms, Canva basics, "
     "and Adobe Illustrator/Photoshop basics for kids."),
    ("Laptop Repairing", "laptop-repairing", "Vocational Training Courses", "3 months",
     "Laptop Repairing."),
    ("3D Animation", "3d-animation", "Designing", "6 months",
     "Computer animation: the process used for digitally generating animations, covering both "
     "static scenes (CGI) and moving images."),
    ("CCTV Camera System Installation", "cctv-camera-system-installation",
     "Vocational Training Courses", "3 months", "CCTV Camera Installation."),
    ("Python Programming", "python-programming", "Development", "3 months",
     "An introduction to programming and the Python language, covering core programming concepts "
     "such as data structures, conditionals, loops, variables, and functions."),
    ("AI & Chatbot", "ai-and-chatbot", "Development", "5 months",
     "Building a chatbot -- computer software that conducts an online chat conversation via text "
     "or text-to-speech in place of providing direct contact with a live human agent."),
    ("Microsoft Dynamic CRM", "microsoft-dynamic-crm", "Development", "4 months",
     "Microsoft Dynamic CRM."),
    ("English Language Course", "ELC", "Language", "3 months", "English Language Course."),
    ("Shopify E-Commerce Expert", "shopify-e-commerce-expert", "Entrepreneurship", "3 months",
     "Shopify E-Commerce Expert."),
    ("IT Professional", "it-professional", "Networking", "3 months",
     "A comprehensive program covering a range of IT topics to provide a strong foundation for "
     "entry-level IT industry positions."),
    ("Cisco Certified Support Technician -- Cyber Security",
     "cisco-certified-support-technician-cyber-security", "Networking", "3 months",
     "Cisco Certified Support Technician Cyber Security."),
    ("Cisco Certified Support Technician -- Networking",
     "cisco-certified-support-technician-networking", "Networking", "3 months",
     "Cisco Certified Support Technician Networking."),
    ("Cyber Security Professional", "cyber-security-professional", "Networking", "6 months",
     "Cyber Security Professional."),
    ("Skill Accelerator Bootcamp", "skill-accelerator-bootcamp", "Development", "4 months",
     "Skill Accelerator Bootcamp."),
    ("The Fintech Blueprint", "the-fintech-blueprint", "Development", "3 months",
     "The Fintech Blueprint."),
    ("Microsoft Dynamic Finance & Operations", "microsoft-dynamic-finance-and-operations",
     "Development", "4 months", "Microsoft Dynamic Finance & Operations."),
    ("Certified Blockchain Developer", "certified-blockchain-developer", "Blockchain", "6 months",
     "Blockchain Developer Track, focused on Solidity."),
    ("Flow HCM Certified HR Professional", "flow-hcm-certified-hr-professional", "Development",
     "3 months", "Flow HCM Certified HR Professional."),
    ("Freelancing", "freelancing", "Entrepreneurship", "8 months",
     "Fundamentals of computers, digital marketing (online strategies, SEO, social media), "
     "LinkedIn marketing, freelance platforms, business development, and project management."),
    ("Turkish Language Course", "turkish-language-course", "Language", "3 months",
     "Turkish Language Course."),
    ("Game Development", "game-development", "Development", "6 months", "Game Development."),
    ("Snowflake SnowPro Core Certification", "snowflake-snow-pro-core-certification", "Development",
     "4 months", "Snowflake SnowPro Core Certification."),
    ("Amazon FBA Mastery", "amazon-fba-mastery", "Entrepreneurship", "3 months",
     "Amazon FBA Mastery."),
    ("Security Professional", "security-professional", "Networking", "4 months",
     "Security Professional."),
    ("Network Professional", "network-professional", "Networking", "6 months", "CCNA + CCNP."),
]

# --- Full campus network (name, url, city, address, maps_link or None) ---
CAMPUSES = [
    ("Head Office", "karachi/head-office-bahadurabad", "Karachi", "Bahadurabad, Saylani Head Office 4th floor", "https://www.google.com/maps?q=24.88275,67.06812"),
    ("Numaish Campus", "karachi/numaish-campus", "Karachi", "Opposite to Hussaini Blood Bank, Near Numaish Chowrangi", "https://www.google.com/maps?q=24.8718942,67.0345009"),
    ("Gulshan Campus", "karachi/gulshan-campus", "Karachi", "Mumtaz Mobile Mall, 2nd Floor, Gulshan e Iqbal", "https://www.google.com/maps?q=24.9257039,67.0075662"),
    ("Malir Campus", "karachi/malir-campus", "Karachi", "SB-12 KN Gohar Green City Malir", "https://www.google.com/maps?q=24.866768,67.198313"),
    ("Faisalabad Campus", "faisalabad/faisalabad-campus", "Faisalabad", "Saylani House, 3rd Floor, Chowk, Lal Mill Rd, Factory Area, Faisalabad", None),
    ("Aliabad Female Campus", "karachi/aliabad-female-campus", "Karachi", "Opposite to Tabba Heart Institute Hussainabad", "https://www.google.com/maps?q=24.9203516,66.9812454"),
    ("Paposh Campus", "karachi/paposh-campus", "Karachi", "C 14, 6B, Block 5-C Block 5 D Block 5 Nazimabad, Karachi, 74600", "https://www.google.com/maps?q=24.9212401,67.0262533"),
    ("Sarfraz Campus", "hyderabad/sarfraz-campus", "Hyderabad", "Sarfaraz Chowrangi, Opposite Bin Tayyab Hospital", "https://www.google.com/maps?q=25.3973517,68.3643832"),
    ("Mohsin and Huma Campus", "peshawar/mohsin-and-huma-campus", "Peshawar", "University Town", "https://www.google.com/maps?q=33.9952454,71.4090876"),
    ("Quetta Campus", "quetta/quetta-campus", "Quetta", "Directorate General, Social Welfare Department, Brewery Rd, Quetta", "https://www.google.com/maps?q=30.1913674,66.8745782"),
    ("F-10 Markaz Campus", "islamabad/f-10-markaz-campus", "Islamabad", "Saylani Office F-10 Markaz F 10/3, Islamabad", "https://www.google.com/maps?q=33.6819307,72.9764771"),
    ("Zaitoon Ashraf IT Park", "karachi/zaitoon-ashraf-it-park", "Karachi", "Baloch Colony, Opposite to City School", "https://www.google.com/maps?q=24.8627824,67.0039073"),
    ("SBIL (Saylani School of Business and Islamic Leadership)", "karachi/sbil-saylani-school-of-business-and-islamic-leadership", "Karachi", "Gulshan Chowrangi, Mumtaz Mobile Mall", "https://www.google.com/maps?q=24.9258152,67.0075394"),
    ("Memon Society Campus", "hyderabad/memon-society-campus", "Hyderabad", "Memon Society, Phase 1, Qasimabad, Hyderabad, Sindh", None),
    ("Rawalpindi Campus", "rawalpindi/rawalpindi-campus", "Rawalpindi", "S 25-A, 4th Road, Ashghal Mall Scheme, National Marketing, Near Steps College, Rawalpindi", None),
    ("North Karachi Campus", "karachi/north-karachi-campus", "Karachi", "North Karachi, 2 minutes chowrangi sector 5D, near Imam Bargah, opposite Ali Classic building", None),
    ("Autoban Campus", "hyderabad/autoban-campus", "Hyderabad", "Near Maji Hospital, Auto Bhan Road, Latifabad, Hyderabad", None),
    ("Gujranwala Taj Campus", "gujranwala/gujranwala-taj-campus", "Gujranwala", "2nd Floor, Taj Islamic Research Center, Block-C, Magnolia Park, Near Jamia Masjid Taj, Gujranwala", None),
    ("Saylani Hajiyani Sakeena IT Campus", "karachi/saylani-hajiyani-sakeena-it-campus", "Karachi", "Friends Arcade, Chandni Chowk, Near Stadium Road, Karachi", None),
    ("Saylani TITAN Sukkur Campus", "sukkur/saylani-titan-sukkur-campus", "Sukkur", "Military Road Sukkur, opposite to Hotel One", None),
    ("Mohsin and Huma Campus (Lakki Marwat)", "lakki-marwat/mohsin-and-huma-campus", "Lakki Marwat", "Peshawar road area", None),
    ("Saylani TITAN Zamzama Campus", "karachi/saylani-titan-zamzama-campus", "Karachi", "3rd Floor, Plot No. 15-C, 9th Zamzama Ln, Zamzama Commercial Area, Defence V, Karachi, 75600", None),
    ("SVTI Gulshan Campus", "karachi/svti-gulshan-campus", "Karachi", "Mumtaz Mobile Mall, 2nd Floor, Gulshan e Iqbal", None),
    ("ASF Campus", "karachi/asf-campus", "Karachi", "V4WQ+9J8, Faisal Cantonment, Karachi", None),
    ("DHA Phase 7 Female Campus", "karachi/dha-phase-7-female-campus", "Karachi", "DHA Phase 8", None),
    ("Saylani IT Park Multan", "multan/saylani-it-park-multan", "Multan", "Service road, near new Khan bus terminal, Vehari Chowk, Multan", None),
    ("SMIT Ghotki Campus", "ghotki-/smit-ghotki-campus", "Ghotki", "Jeejal Aman Public School", None),
    ("Turbat Campus", "turbat/turbat-campus", "Turbat", "Turbat, Balochistan", None),
    ("PMA Kakul Campus", "abbottabad/pma-kakul-campus", "Abbottabad", "Pakistan Military Academy, Abbottabad", "https://www.google.com/maps?q=34.18739044703245,73.2602855801596"),
    ("DF Army Public School Campus", "rawalpindi/df-army-public-school", "Rawalpindi", "Rawal Road, Rawalpindi", "https://www.google.com/maps?q=33.60270459342268,73.0873128684933"),
    ("GOC Lhr LGITE 11 Div Campus", "lahore/Campus", "Lahore", "Shalimar Link Road, Ghaziabad, Shalimar Town, Lahore, 05466", None),
    ("Bahria College DHA Phase 2 Campus", "karachi/Bahria-Collage-DHA-Phase 2 Campus", "Karachi", "Sabir SRE, Defence Housing Authority, Karachi", None),
    ("Bahria College Nore 1 Campus", "karachi/Bahria Collage Nore 1 Campus", "Karachi", "Moulvi Tamizuddin Khan Rd, Naval Officers Residential Estate 1, Naval Officers Colony, Karachi", "https://www.google.com/maps?q=24.8146,66.9738"),
    ("Bahria Subh-e-Nau Secondary Campus", "karachi/Bahria Subh-e-Nau Secondary Campus", "Karachi", "Roomi Ave, NORE 1, Karachi, Sindh", None),
    ("Bahria College 1 Majeed SRE Campus", "karachi/Bahria College Majeed SRE", "Karachi", "Navy Colony, Karachi", None),
    ("Bahria College 2 Majeed SRE Campus", "karachi/Bahria College 2 Majeed SRE Campus", "Karachi", "Navy Colony, Karachi", None),
    ("Bahria College KNCRE Agrataj Campus", "karachi/Bahria College KNCRE Agrataj Campus", "Karachi", "G-23 Taj Masjid Rd, Agra Taj Colony, Karachi", None),
    ("Bahria College Hanif SRE Campus", "karachi/Bahria College Hanif SRE Campus", "Karachi", "Karsaz, Faisal Cantonment, Karachi", None),
    ("Bahria College Khalid SRE Campus", "karachi/Bahria College Khalid SRE Campus", "Karachi", "Bahria College Khalid SRE, Keamari, Younus Abad", None),
    ("Bahria Community Centre Campus", "karachi/bahria-community-centre-campus", "Karachi", "Pakistan Navy Community Centre, Majeed SRE", None),
    ("Waziristan Wana Campus", "waziristan/waziristan-wana-campus", "Waziristan", "Online", None),
    ("Wana Campus", "waziristan/Wana-Campus", "Waziristan", "Wana", "https://www.google.com/maps?q=24.8615,67.0099"),
    ("ASF Public School & College Lahore", "lahore/asf-campus-lahore", "Lahore", "Khawaja Road, Walton, Punjab, Askari 5, Lahore", "https://www.google.com/maps?q=31.503,74.348"),
    ("Gujranwala QDPS Campus", "gujranwala/Gjranwala QDPS Campus", "Gujranwala", "Main G.T. Road, Industrial Estate 1, Gujranwala (P.O. Box 52250)", "https://www.google.com/maps?q=32.18522231662376,74.1824166227844"),
    ("ASF Public School Rawalpindi", "rawalpindi/ASF rawalpindi", "Rawalpindi", "Khawaja Road, Walton, Punjab, Askari 5", None),
    ("Mehfooz Shaheed Garrison Campus", "lahore/Mehfooz shaheed garrison Campus", "Lahore", "Lahore Cantonment area", None),
    ("Bahria College Karsaz", "karachi/bahria-karsaz", "Karachi", "V3MX+V6Q, PNS Karsaz Rd, Karsaz, Faisal Cantonment, Karachi", "https://www.google.com/maps?q=24.8419087,67.0042702"),
    ("Balochistan Campus", "balochistan/balochistan-campus", "Balochistan", "Balochistan", None),
    ("Washuk Campus", "balochistan/Washuk Campus", "Balochistan (Washuk)", "Washuk, Balochistan", None),
    ("Kharan Campus", "balochistan/kharan-campus", "Balochistan (Kharan)", "Kharan, Balochistan", None),
]

BASE_COURSE_URL = "https://www.saylanimit.com/courses/"
BASE_CAMPUS_URL = "https://www.saylanimit.com/campuses/"


def _write(source_url: str, title: str, text: str) -> None:
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    record = {
        "source_url": source_url,
        "title": title,
        "text": text,
        "source_type": "official_website",
        "crawl_date": NOW,
        "content_hash": content_hash,
    }
    filename = hashlib.sha256(source_url.encode("utf-8")).hexdigest()[:16] + ".json"
    (OUT_DIR / filename).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    written = 0

    for name, slug, category, duration, description in COURSES:
        url = BASE_COURSE_URL + slug
        text = (
            f"Course: {name}. Category: {category}. Duration: {duration}. "
            f"Description: {description} "
            f"This course is offered free of cost by Saylani Mass IT Training (SMIT), an "
            f"initiative of Saylani Welfare International Trust. Admission is via the online "
            f"registration form at https://www.saylanimit.com/enroll, selecting this course and "
            f"a preferred campus/city.{FEED_NOTE}"
        )
        _write(url, f"{name} | Saylani Mass IT Training", text)
        written += 1

    for name, slug, city, address, maps_link in CAMPUSES:
        url = BASE_CAMPUS_URL + slug
        maps_sentence = f" Google Maps location: {maps_link}." if maps_link else ""
        text = (
            f"SMIT Campus: {name}, located in {city}, Pakistan. Address: {address}.{maps_sentence} "
            f"This is one of Saylani Mass IT Training's (SMIT) training centers where free IT "
            f"courses are delivered on-site. To check current course availability at this campus "
            f"or enroll, visit https://www.saylanimit.com/enroll and select {city} as the city.{FEED_NOTE}"
        )
        _write(url, f"{name}, {city} | SMIT Campus Location & Details", text)
        written += 1

    # --- Aggregate index documents, for "how many / which" style questions ---
    course_names_by_category: dict[str, list[str]] = {}
    for name, _slug, category, duration, _desc in COURSES:
        course_names_by_category.setdefault(category, []).append(f"{name} ({duration})")

    courses_index_text = (
        f"Saylani Mass IT Training (SMIT) currently offers {len(COURSES)} IT and vocational "
        f"courses in total, almost all free of cost, organized into these categories:\n\n"
    )
    for category, names in course_names_by_category.items():
        courses_index_text += f"{category} ({len(names)} courses): " + "; ".join(names) + ".\n"
    courses_index_text += (
        f"\nFull course list and details: https://www.saylanimit.com/courses. To enroll in any "
        f"course: https://www.saylanimit.com/enroll.{FEED_NOTE}"
    )
    _write(
        "https://www.saylanimit.com/courses#full-index",
        "Complete Course Catalog Index | Saylani Mass IT Training",
        courses_index_text,
    )
    written += 1

    campuses_by_city: dict[str, list[str]] = {}
    for name, _slug, city, _address, _maps in CAMPUSES:
        campuses_by_city.setdefault(city, []).append(name)

    campuses_index_text = (
        f"Saylani Mass IT Training (SMIT) operates {len(CAMPUSES)} campuses across Pakistan, in "
        f"the following cities/areas: {', '.join(sorted(campuses_by_city.keys()))}.\n\n"
    )
    for city, names in campuses_by_city.items():
        campuses_index_text += f"{city}: " + "; ".join(names) + ".\n"
    campuses_index_text += (
        f"\nFull campus list with addresses and maps: https://www.saylanimit.com/campuses. "
        f"Students can select their nearest campus during enrollment at "
        f"https://www.saylanimit.com/enroll.{FEED_NOTE}"
    )
    _write(
        "https://www.saylanimit.com/campuses#full-index",
        "Complete Campus Network Index | Saylani Mass IT Training",
        campuses_index_text,
    )
    written += 1

    print(f"Wrote {written} SMIT course/campus documents to {OUT_DIR}")
    print(f"  - {len(COURSES)} individual course pages")
    print(f"  - {len(CAMPUSES)} individual campus pages")
    print("  - 2 aggregate index pages (courses, campuses)")
    return written


if __name__ == "__main__":
    main()
