import re
import os

def main():
    base_dir = r"c:\Users\bhara\Downloads\Final_AG"
    main_py_path = os.path.join(base_dir, "main.py")
    enhanced_py_path = os.path.join(base_dir, "Enhanced_UI_main.py")
    
    with open(main_py_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 1. 3D Script & Canvas
    text = text.replace(
        '<script src="https://cdn.jsdelivr.net/npm/gsap@3.12.5/dist/ScrollTrigger.min.js"></script>',
        '<script src="https://cdn.jsdelivr.net/npm/gsap@3.12.5/dist/ScrollTrigger.min.js"></script>\n    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>'
    )
    text = text.replace(
        '<div class="aurora">',
        '<canvas id="bg-canvas" style="position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; pointer-events: none; z-index: 0; opacity: 0.6;"></canvas>\n    <div class="aurora">'
    )

    init_three_js = """
        function initThreeJS() {
            const canvas = document.getElementById('bg-canvas');
            const scene = new THREE.Scene();
            const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
            const renderer = new THREE.WebGLRenderer({ canvas: canvas, alpha: true, antialias: true });
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(window.devicePixelRatio);

            const group = new THREE.Group();
            
            const createCross = () => {
                const shape = new THREE.Shape();
                const d = 0.5, w = 1.5;
                shape.moveTo(-d, d); shape.lineTo(-w, d); shape.lineTo(-w, -d); shape.lineTo(-d, -d);
                shape.lineTo(-d, -w); shape.lineTo(d, -w); shape.lineTo(d, -d); shape.lineTo(w, -d);
                shape.lineTo(w, d); shape.lineTo(d, d); shape.lineTo(d, w); shape.lineTo(-d, w);
                
                const extrudeSettings = { depth: 0.5, bevelEnabled: true, bevelSegments: 2, steps: 1, bevelSize: 0.1, bevelThickness: 0.1 };
                const geometry = new THREE.ExtrudeGeometry(shape, extrudeSettings);
                geometry.center();
                
                const material = new THREE.MeshPhysicalMaterial({
                    color: 0x66e3ff, metalness: 0.2, roughness: 0.2,
                    transmission: 0.9, transparent: true, opacity: 0.35, wireframe: false,
                    clearcoat: 1.0, clearcoatRoughness: 0.1
                });
                return new THREE.Mesh(geometry, material);
            };

            for(let i=0; i<15; i++) {
                const mesh = createCross();
                mesh.position.set((Math.random() - 0.5) * 40, (Math.random() - 0.5) * 40, (Math.random() - 0.5) * 20 - 10);
                mesh.rotation.set(Math.random() * Math.PI, Math.random() * Math.PI, Math.random() * Math.PI);
                const scale = Math.random() * 0.8 + 0.4;
                mesh.scale.set(scale, scale, scale);
                group.add(mesh);
            }
            scene.add(group);

            const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
            scene.add(ambientLight);
            const directionalLight = new THREE.DirectionalLight(0x66e3ff, 2);
            directionalLight.position.set(1, 1, 2);
            scene.add(directionalLight);
            
            const light2 = new THREE.DirectionalLight(0xff6e9a, 1.5);
            light2.position.set(-1, -1, 1);
            scene.add(light2);

            camera.position.z = 10;

            function animate() {
                requestAnimationFrame(animate);
                group.rotation.x += 0.0005;
                group.rotation.y += 0.001;
                renderer.render(scene, camera);
            }
            animate();

            if (window.gsap && window.ScrollTrigger) {
                gsap.to(group.rotation, {
                    y: Math.PI * 2, x: Math.PI,
                    scrollTrigger: { trigger: "body", start: "top top", end: "bottom bottom", scrub: 1 }
                });
                gsap.to(group.position, {
                    y: 5, z: 2,
                    scrollTrigger: { trigger: "body", start: "top top", end: "bottom bottom", scrub: true }
                });
            }

            window.addEventListener('resize', () => {
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
            });
        }
"""
    text = text.replace('function initScrollAnimations() {', init_three_js + '\n        function initScrollAnimations() {')
    text = text.replace('initControls();', 'initControls();\n            initThreeJS();')

    # 2. Margins and Overlaps
    text = re.sub(
        r'\.control-ribbon\s*{\s*([^}]*?)margin-top:\s*-72px;([^}]*?)}', 
        r'.control-ribbon { \1margin-top: -20px; z-index: 10; position: relative;\2}', 
        text
    )
    text = text.replace('margin=dict(l=28, r=28, t=70, b=38)', 'margin=dict(l=45, r=45, t=85, b=55)')

    # 3. High Risk Pill CSS
    high_risk_css = """
        .badge-high {
            background: linear-gradient(135deg, #ff0044 0%, #ff4b2b 100%);
            color: #ffffff !important;
            font-weight: 900;
            padding: 8px 16px;
            border-radius: 999px;
            display: inline-block;
            box-shadow: 0 0 20px rgba(255, 0, 68, 0.8), 0 0 40px rgba(255, 75, 43, 0.6);
            border: 2px solid rgba(255, 179, 179, 0.5);
            text-shadow: 0 0 10px rgba(255,255,255,0.8);
            animation: pulse-high-risk 1.5s infinite;
        }
        .badge-medium {
            background: linear-gradient(135deg, #d97706 0%, #f59e0b 100%);
            color: #1f2937;
            padding: 8px 16px;
            border-radius: 999px;
            display: inline-block;
            font-weight: 900;
            box-shadow: 0 8px 20px rgba(217, 119, 6, 0.4);
        }
        .badge-low {
            background: linear-gradient(135deg, #059669 0%, #10b981 100%);
            color: #ffffff;
            padding: 8px 16px;
            border-radius: 999px;
            display: inline-block;
            font-weight: 900;
            box-shadow: 0 8px 20px rgba(5, 150, 105, 0.4);
        }
        @keyframes pulse-high-risk {
            0% { box-shadow: 0 0 20px rgba(255, 0, 68, 0.8), 0 0 40px rgba(255, 75, 43, 0.6); }
            50% { box-shadow: 0 0 30px rgba(255, 0, 68, 1), 0 0 60px rgba(255, 75, 43, 0.8); transform: scale(1.02); }
            100% { box-shadow: 0 0 20px rgba(255, 0, 68, 0.8), 0 0 40px rgba(255, 75, 43, 0.6); }
        }
        """
    text = text.replace('</style>', high_risk_css + '</style>')

    # 4. JSON box redesign
    json_old_pattern = r'<div style="background: rgba\(10, 14, 26, 0\.6\); border: 1px solid rgba\(6, 182, 212, 0\.2\); border-radius: 10px; padding: 20px; font-family: \'Courier New\', monospace;">\s*<pre style="color: #06b6d4; font-size: 0\.95em; line-height: 1\.6; margin: 0; white-space: pre-wrap;">{{'
    json_new = '''<div style="background: linear-gradient(180deg, rgba(8, 12, 23, 0.9), rgba(4, 7, 13, 0.95)); border: 1px solid rgba(102, 227, 255, 0.3); border-radius: 12px; padding: 25px; font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace; position: relative; overflow: hidden; box-shadow: inset 0 0 20px rgba(102, 227, 255, 0.05), 0 10px 30px rgba(0,0,0,0.5);">
                <div style="position: absolute; top: 0; left: 0; width: 100%; height: 4px; background: linear-gradient(90deg, #66e3ff, #7c8cff, #ff6e9a); opacity: 0.8;"></div>
                <div style="display: flex; gap: 8px; margin-bottom: 15px;">
                    <span style="width: 12px; height: 12px; border-radius: 50%; background: #ff5f56; box-shadow: 0 0 5px #ff5f56;"></span>
                    <span style="width: 12px; height: 12px; border-radius: 50%; background: #ffbd2e; box-shadow: 0 0 5px #ffbd2e;"></span>
                    <span style="width: 12px; height: 12px; border-radius: 50%; background: #27c93f; box-shadow: 0 0 5px #27c93f;"></span>
                </div>
                <pre style="color: #66e3ff; font-size: 0.95em; line-height: 1.6; margin: 0; white-space: pre-wrap; text-shadow: 0 0 5px rgba(102, 227, 255, 0.4);">{{'''
    text = re.sub(json_old_pattern, json_new, text)

    # 5. Timeline Function Redesign
    old_timeline_pattern = r'def generate_action_timeline\(patient\):.*?return timeline_html'
    
    new_timeline_func = '''def generate_action_timeline(patient):
    if patient["risk_level"] == "HIGH":
        actions = [
            ("Immediate", "Activate high-risk discharge protocol", "#ef4444"),
            ("24 Hours", "First follow-up contact via phone call", "#f97316"),
            ("48 Hours", "In-person clinic visit or home health assessment", "#fbbf24"),
            ("Week 1", "Medication reconciliation and adherence check", "#facc15"),
        ]
    elif patient["risk_level"] == "MEDIUM":
        actions = [
            ("24-48 Hours", "Standard discharge follow-up call", "#fbbf24"),
            ("Week 1", "Primary care appointment scheduled", "#facc15"),
            ("Week 2", "Case manager check-in call", "#a3e635"),
            ("Month 1", "Care plan review & adjustment", "#22c55e"),
        ]
    else:
        actions = [
            ("Week 1", "Routine follow-up call", "#a3e635"),
            ("2 Weeks", "Primary care visit", "#22c55e"),
            ("Month 1", "Standard monitoring per plan", "#10b981"),
        ]

    timeline_html = "<div style='display: flex; flex-direction: column; gap: 15px; position: relative; padding-left: 20px;'>"
    timeline_html += "<div style='position: absolute; left: 6px; top: 10px; bottom: 10px; width: 2px; background: rgba(255,255,255,0.1);'></div>"
    for i, (timeframe, action, color) in enumerate(actions):
        timeline_html += f"""
        <div style="position: relative; display: flex; align-items: center; gap: 15px; background: rgba(255,255,255,0.03); padding: 15px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05); transition: transform 0.2s; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <div style="position: absolute; left: -21px; top: 50%; transform: translateY(-50%); width: 14px; height: 14px; border-radius: 50%; background: {color}; box-shadow: 0 0 10px {color}; border: 2px solid #000;"></div>
            <div style="min-width: 100px; font-weight: bold; color: {color}; text-transform: uppercase; font-size: 0.85em; letter-spacing: 0.05em;">{timeframe}</div>
            <div style="color: #f9fafb; font-size: 1.05em;">{action}</div>
        </div>
        """
    timeline_html += "</div>"
    return timeline_html'''
    
    text = re.sub(old_timeline_pattern, new_timeline_func, text, flags=re.DOTALL)

    # 6. Copywriting modifications
    
    text = re.sub(
        r'This experience keeps the original data preparation, logistic regression model,\s*and Mosaic AI reasoning intact, while upgrading the presentation layer into a cinematic,\s*judge-friendly risk command center\.',
        r'This enterprise-grade predictive analytics engine harnesses advanced logistic regression and Mosaic AI reasoning.\n                            It transforms raw clinical history into an interactive, 3D command center, surfacing precise readmission risks with full model transparency and actionable insights.',
        text
    )

    text = re.sub(
        r'`This experience keeps the original <strong>data preparation</strong>, <strong>logistic regression model</strong>, and ` \+\s*`<strong>Mosaic AI reasoning</strong> intact, while upgrading the presentation layer into a cinematic, judge-friendly risk command center\. `',
        r'`This enterprise-grade predictive analytics engine harnesses advanced logistic regression and <strong>Mosaic AI reasoning</strong>. ` + \n            `It transforms raw clinical history into an interactive, 3D command center, surfacing precise readmission risks with full model transparency and actionable insights. `',
        text
    )

    text = re.sub(
        r'Search any patient ID from the test results, or jump straight into a prepared high-risk or low-risk example\.\s*The charts, model evidence, and Mosaic AI recommendations update together\.',
        r'Instantly query any patient ID from the evaluation cohort, or initialize a pre-configured high-risk/low-risk clinical scenario.\n                    Interactive charts, logistic model evidence, and AI directives synchronize in real-time.',
        text
    )

    text = re.sub(
        r'As we scroll, the narrative steps and chart stage stay in sync, making the case easy to explain live\s*without losing clinical detail or model transparency\.',
        r'Scroll seamlessly through the patient\'s diagnostic profile. The synchronized narrative and advanced visual stage ensure that clinical nuance and model transparency are conveyed with maximum precision.',
        text
    )

    with open(enhanced_py_path, "w", encoding="utf-8") as f:
        f.write(text)

if __name__ == "__main__":
    main()
