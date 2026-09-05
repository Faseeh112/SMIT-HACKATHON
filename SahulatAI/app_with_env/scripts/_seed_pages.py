"""
One-off helper: writes a set of manually-fetched, real page contents from
saylaniwelfare.com and saylanimit.com into knowledge_base/raw/ in the same
JSON schema crawl_saylani.py produces, so ingest_documents.py can index them
immediately without needing network access to the live sites.

This is a bootstrap/seed set, not a replacement for scripts/crawl_saylani.py
-- re-run the real crawler later to pick up new/changed pages.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "knowledge_base" / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NOW = datetime.now(timezone.utc).isoformat()

PAGES = [
    {
        "source_url": "https://saylaniwelfare.com/",
        "title": "Saylani Welfare | Non Governmental Organization in Pakistan",
        "text": (
            "Saylani Welfare International Trust is a Pakistani non-governmental "
            "organization (NGO) headquartered at A-25, Bahadurabad Chowrangi, "
            "Karachi, Pakistan. Saylani Welfare is on the ground and already "
            "working with local communities to assess how best to support "
            "underprivileged families in more than 63 areas of day to day life, "
            "including food, health, education, social welfare, clean water, "
            "marriage assistance, mass IT training, vocational training, "
            "pilgrim services, school fees, easy loans, medical and diagnostic "
            "services, and disaster relief.\n\n"
            "Contact: UAN +92 21 111 729 526, WhatsApp 0311 1729526, "
            "email info@saylaniwelfare.com. "
            "USA: +1 337 337 2370. UK: +44 115 970 6256. Canada: +1 888 572 3485.\n\n"
            "The main site sections are: Home, About, Services, Media, Contact Us, "
            "Bank Details, and Donor Dashboard. Donation categories include Sadqa "
            "e Jariah, Sadqa/Aqiqah Animal, Food Donation, Education, and Medical "
            "& Healthcare."
        ),
    },
    {
        "source_url": "https://saylaniwelfare.com/about/saylani-introduction",
        "title": "About Saylani Welfare International Trust",
        "text": (
            "Saylani Welfare International Trust was established in May 1999 by "
            "spiritual and religious scholar Maulana Bashir Farooqui (also "
            "referred to as Maulana Bashir Ahmed Farooqui), headquartered in "
            "Bahadurabad, Karachi, Pakistan. It is one of Pakistan's largest "
            "NGOs, focusing on breaking the cycle of poverty and helping the "
            "underprivileged through food distribution, healthcare, education, "
            "clean water, vocational and IT training, marriage assistance, "
            "pilgrim services, and disaster relief. Saylani spends billions of "
            "rupees annually on humanitarian services and reaches hundreds of "
            "thousands of people daily through its network of centers across "
            "Pakistan, with additional offices supporting fundraising in the "
            "USA, UK, and Canada."
        ),
    },
    {
        "source_url": "https://saylaniwelfare.com/services/education/technical-education/saylani-mass-it-training",
        "title": "Saylani Mass IT-Training (SMIT) | Saylani Welfare",
        "text": (
            "Saylani Mass IT Training (SMIT) is a flagship project of Saylani "
            "Welfare to impart free, industry-relevant technology education to "
            "the youth of Pakistan. Founded in 2013, SMIT provides students "
            "with hands-on digital skills for careers in the global technology "
            "sector. SMIT offers over 80 specialized courses, most delivered "
            "free of cost; students are only expected to have basic computer "
            "skills. Courses are delivered by experts in a hands-on, "
            "project-based way. SMIT has trained over 200,000 students to date. "
            "Who it serves: students, graduates and promising professionals. "
            "Service coverage: available throughout Pakistan. Availability: "
            "year round, free & scheduled. Contact: +92 21 111 729 526.\n\n"
            "How the service works: (1) Identify Need — students are evaluated "
            "based on academic background, interests, and career interest prior "
            "to choosing appropriate technology courses. (2) Deliver Service — "
            "technical education with focus on industry-relevant practical "
            "training, mentorship, projects, certifications, and career-focused "
            "education by industry experts. (3) Follow Up — ongoing mentorship, "
            "portfolio advice, industry exposure, and career support after "
            "course completion."
        ),
    },
    {
        "source_url": "https://saylaniwelfare.com/contact-us",
        "title": "Contact Us | Saylani Welfare",
        "text": (
            "Saylani Welfare International Trust head office: A-25, Bahadurabad "
            "Chowrangi, Karachi, Pakistan. Phone/UAN: +92 21 111 729 526. "
            "Alternate numbers: +92 21 38729526, +92 311 1729526 (WhatsApp/cell), "
            "+1 337 337 2370 (USA), +44 115 970 6256 (UK), +1 888 572 3485 "
            "(Canada). Email: info@saylaniwelfare.com."
        ),
    },
    {
        "source_url": "https://www.saylanimit.com/",
        "title": "Saylani Mass IT Training – IT Education in Pakistan",
        "text": (
            "Saylani Mass IT Training (SMIT), an initiative of Saylani Welfare "
            "International Trust, is Pakistan's largest non-profit IT training "
            "provider, serving since 2013. SMIT offers 100% free education "
            "(no tuition fees), hands-on training with real-world projects, a "
            "reported 70% employment & freelancing success rate among "
            "graduates, and has supported 150+ startups launched globally. "
            "SMIT is recognized by Cisco and other global tech partners. "
            "SMIT's vision is training 10 million IT experts to help power "
            "Pakistan's growing digital economy.\n\n"
            "Course categories available: Development (e.g. Odoo Functional "
            "Consultant, DevOps Engineer, Mobile App Development with React "
            "Native, Generative AI & Chatbot), Designing (Video Content "
            "Creation, Video Animation, UI/UX Design with AI, AutoCAD), "
            "Networking (IT Support Engineer, SOC Analyst/CyberOps Associate, "
            "Networking Essentials, Cybersecurity Essentials), Entrepreneurship "
            "(AI & Game Creators, Little Geniuses coding/design/AI fun lab for "
            "kids, Certified AI & Digital Assets Engineer, Office Automation), "
            "and Vocational Training (Health Safety and Environment/HSE, Fire "
            "Alarm System Installation, Domestic Electrician, Plumber "
            "Technician).\n\n"
            "Popular course durations: DevOps Engineer (9 months), Mobile App "
            "Development React Native (5 months), Video Content Creation (4 "
            "months), AI & Game Creators (2 months). To see the full course "
            "catalog: https://www.saylanimit.com/courses. To apply: "
            "https://www.saylanimit.com/enroll. Head office: A-25, Bahadurabad "
            "Chowrangi, Karachi, Pakistan. Phone: +92 21 111 729 526. Email: "
            "saylanimass@gmail.com."
        ),
    },
    {
        "source_url": "https://www.saylanimit.com/about",
        "title": "About Us | Saylani Mass IT Training",
        "text": (
            "SMIT, an initiative of Saylani Welfare International Trust, is "
            "dedicated to providing high-quality, modern IT training to "
            "underprivileged and aspiring students. Its mission is to equip the "
            "next generation with coding and digital skills necessary to "
            "thrive in the 21st-century economy, irrespective of financial "
            "background. SMIT believes access to top-tier technological "
            "education is a right, not a privilege, and focuses intensely on "
            "practical application and job readiness so graduates are prepared "
            "to compete in the global marketplace."
        ),
    },
    {
        "source_url": "https://www.saylanimit.com/courses",
        "title": "Explore IT Courses | Saylani Mass IT Training",
        "text": (
            "SMIT's course catalog is organized into categories:\n\n"
            "Development: Odoo Functional Consultant, DevOps Engineer, Mobile "
            "App Development (React Native), Generative AI & Chatbot.\n"
            "Designing: Video Content Creation, Video Animation, UI/UX Design "
            "With AI, AutoCAD.\n"
            "Networking: IT Support Engineer, SOC Analyst/CyberOps Associate, "
            "Networking Essentials, Cybersecurity Essentials.\n"
            "Entrepreneurship: AI & Game Creators, Little Geniuses (coding, "
            "design and AI fun lab for children), Certified AI & Digital "
            "Assets Engineer (for scholars and finance professionals), Office "
            "Automation.\n"
            "Vocational Training: Health, Safety and Environment (HSE), Fire "
            "Alarm System Installation, Domestic Electrician, Plumber "
            "Technician.\n\n"
            "Other courses referenced on the site include Web Development "
            "(Modern Web Application Development), Agentic AI, Artificial "
            "Intelligence and Data Science, Graphic Designing with AI, and 3D "
            "Animation. To enroll in any course, visit "
            "https://www.saylanimit.com/enroll."
        ),
    },
    {
        "source_url": "https://www.saylanimit.com/courses/mobile-app-development-react-native",
        "title": "Mobile App Development (React Native) | Saylani Mass IT Training",
        "text": (
            "Course: Mobile App Development (React Native). Duration: 5 "
            "months. Description: designed to equip learners with the skills "
            "required to build modern, high-performance mobile applications "
            "for both Android and iOS using a single codebase. Participants "
            "learn React Native fundamentals, UI/UX development, navigation, "
            "state management, API integration, database connectivity, "
            "authentication, and mobile app deployment through hands-on "
            "projects. This course includes: certificate of completion, "
            "lifetime mentorship & support, and job placement guidance. "
            "Admission is open; class preference is onsite; open to male and "
            "female applicants; offered at multiple Karachi campuses "
            "including Zaitoon Ashraf IT Park, Head Office Bahadurabad, "
            "Aliabad Female Campus, Paposh Campus, Saylani TITAN Zamzama "
            "Campus, Saylani Hajiyani Sakeena IT Campus, North Karachi Campus, "
            "DHA Phase 7 Female Campus, Malir Campus, and Gulshan Campus. "
            "SMIT overall: 200,000+ students trained, 50+ IT courses, 13+ "
            "years of experience, education is 100% free (no tuition fees)."
        ),
    },
    {
        "source_url": "https://www.saylanimit.com/enroll",
        "title": "Enroll Now | Saylani Mass IT Training - IT Courses",
        "text": (
            "To enroll in an SMIT course, applicants fill out the online "
            "Registration Form at https://www.saylanimit.com/enroll, which "
            "asks for: Location & Course Details (country, class preference, "
            "gender, city, course, campus); Personal Information (full name, "
            "father's name, date of birth, email, phone, father's phone, ID "
            "number/CNIC, father's ID number, address); Education & Technical "
            "Details (computer proficiency, last qualification, how you heard "
            "about SMIT, whether you own a laptop); and an uploaded recent "
            "passport-size picture (white or blue background, under 1MB, "
            "jpg/jpeg/png, face clearly visible without glasses).\n\n"
            "Terms applicants must accept: all information provided must be "
            "true and accurate; applicants agree to abide by SMIT's rules, "
            "regulations and policies; applicants must maintain good conduct "
            "and focus solely on learning, avoiding political, unethical, or "
            "unrelated activities (violation may cancel admission); "
            "applicants agree to complete any assigned capstone/project as "
            "part of course requirements; female students are required to "
            "wear an abaya or hijab while attending classes.\n\n"
            "Other admission-related pages: Download ID Card "
            "(https://www.saylanimit.com/download-id-card), Entry Test Status "
            "(https://www.saylanimit.com/entry-test-status), and Check Result "
            "(https://www.saylanimit.com/result)."
        ),
    },
    {
        "source_url": "https://www.saylanimit.com/contact",
        "title": "Contact Us | Saylani Mass IT Training",
        "text": (
            "Saylani Mass IT Training contact details: Address — A-25, "
            "Bahadurabad Chowrangi, Karachi, Pakistan. Phone: +92 21 111 729 "
            "526. Email: saylanimass@gmail.com. Social media: Facebook "
            "(facebook.com/saylani.smit), X/Twitter (@OfficialSwit), "
            "Instagram (instagram.com/saylani.smit), LinkedIn "
            "(linkedin.com/company/saylanimassit)."
        ),
    },
]


def main() -> int:
    written = 0
    for page in PAGES:
        text = page["text"]
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        record = {
            "source_url": page["source_url"],
            "title": page["title"],
            "text": text,
            "source_type": "official_website",
            "crawl_date": NOW,
            "content_hash": content_hash,
        }
        filename = hashlib.sha256(page["source_url"].encode("utf-8")).hexdigest()[:16] + ".json"
        out_path = OUT_DIR / filename
        out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        written += 1
    print(f"Wrote {written} seed page(s) to {OUT_DIR}")
    return written


if __name__ == "__main__":
    main()
