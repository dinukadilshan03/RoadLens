import Image from "next/image";
import Link from "next/link";
import homeImage from "../../public/home_image.png";
import logo from "../../public/logo.png";

const capabilities = [
  { id: "01", label: "Detect", title: "Find the damage.", text: "Locate potholes and surface cracks in road images. Review each detection before taking the next step.", image: "01_detect_road_damage.png", alt: "Road damage illustration with yellow detection boxes around potholes and cracks", href: "/c1", action: "Open damage detection" },
  { id: "02", label: "Segment", title: "See every detail.", text: "Trace damaged surfaces with pixel-level crack masks, using the regions identified during detection.", image: "02_segment_cracks_potholes.png", alt: "Road damage illustration with red pothole masks and blue crack segmentation", href: "/c2", action: "Open damage segmentation" },
  { id: "03", label: "Measure", title: "Understand the impact.", text: "Use segmented damage to assess severity and estimate physical dimensions for a closer condition review.", image: "03_measure_damage_geometry.png", alt: "Illustration of pothole and crack geometry with width and depth measurements", href: "/c3", action: "Open surface intelligence" },
  { id: "04", label: "Assess", title: "Read the road ahead.", text: "Turn vehicle motion and GPS data into segment-level condition assessments to understand ride quality.", image: "04_assess_road_condition.png", alt: "Illustration of a sensor-equipped vehicle and survey crew assessing road condition", href: "/c4", action: "Open road condition" },
];

export default function Home() {
  return (
    <div className="road-home">
      <header className="home-header">
        <Link className="home-brand" href="/" aria-label="RoadLense home"><Image className="home-logo" src={logo} alt="RoadLense" sizes="(max-width: 480px) 160px, 250px" /></Link>
        <nav aria-label="Main navigation"><Link href="/c1">C1 · Damage detection</Link><Link href="/c2">C2 · Damage segmentation</Link><Link href="/c3">C3 · Surface intelligence</Link><Link href="/c4">C4 · Road condition</Link></nav>
        <Link className="home-button small-button" href="/c1">Start an assessment <span aria-hidden="true">↗</span></Link>
      </header>
      <main>
        <section className="home-hero home-hero-panorama" aria-labelledby="hero-title">
          <div className="hero-copy">
            <p className="home-eyebrow"><span className="status-dot" /> ROAD CONDITION INTELLIGENCE</p>
            <h1 id="hero-title">Better roads start with<br /><em>a closer look.</em></h1>
            <p className="hero-description">From the first crack to the way a road feels. Bring surface damage and ride quality into one clear picture.</p>
            <div className="hero-actions"><Link className="home-button" href="/c1">C1 · Damage detection <span aria-hidden="true">→</span></Link><Link className="home-button home-button-secondary" href="/c2">C2 · Damage segmentation <span aria-hidden="true">→</span></Link><Link className="home-button home-button-secondary" href="/c3">C3 · Surface intelligence <span aria-hidden="true">→</span></Link><Link className="home-button home-button-secondary" href="/c4">C4 · Road condition <span aria-hidden="true">→</span></Link></div>
            <p className="hero-footnote"><span aria-hidden="true">+</span> Image-based inspection. Sensor-based insight.</p>
          </div>
          <div className="hero-road-scene">
            <div className="road-scene-halo" aria-hidden="true" />
            <Image src={homeImage} alt="Illustration of road survey crews inspecting damaged pavement beside a sensor-equipped vehicle scanning potholes and cracks" preload sizes="(max-width: 760px) 100vw, 1280px" />
            <div className="road-scene-caption"><span>ON THE SURFACE</span><span className="road-scene-dash" aria-hidden="true" /><span>A CLEARER PICTURE OF EVERY JOURNEY</span></div>
          </div>
        </section>
        <div className="home-overview"><p>A clearer view of<br /><strong>road condition.</strong></p><div><span>01—03</span><p>Surface damage<br />& severity</p></div><div><span>04</span><p>Road roughness<br />& ride quality</p></div><a href="#workflow">Connected by design <span aria-hidden="true">↗</span></a></div>
        <section className="capabilities-section" id="capabilities" aria-labelledby="capabilities-title">
          <div className="section-heading"><div><p className="home-eyebrow">THE TOOLKIT</p><h2 id="capabilities-title">Every angle of the road.</h2></div><p>Four connected capabilities.<br />One more informed assessment.</p></div>
          <div className="capability-grid">{capabilities.map((item) => <article className="capability-card" key={item.id}>
            <div className="card-image"><Image src={`/${item.image}`} alt={item.alt} fill sizes="(max-width: 600px) 100vw, (max-width: 1000px) 50vw, 25vw" /><span className="card-number">{item.id} / {item.label}</span></div>
            <div className="card-copy"><h3>{item.title}</h3><p>{item.text}</p>{item.href ? <Link href={item.href}>{item.action}<span aria-hidden="true">↗</span></Link> : <span className="pipeline-label">{item.action}</span>}</div>
          </article>)}</div>
        </section>
        <section className="workflow-section" id="workflow" aria-labelledby="workflow-title">
          <div className="workflow-intro"><p className="home-eyebrow">FROM OBSERVATION TO INSIGHT</p><h2 id="workflow-title">Make the surface<br />make sense.</h2><p>Start with the data you have.<br />Follow the details that matter.</p><Link className="home-text-link" href="/c1">Start your inspection <span aria-hidden="true">→</span></Link></div>
          <ol className="workflow-steps"><li><span>01</span><div><h3>Bring your road into view</h3><p>Upload a road image for damage detection, or use motion sensor data to assess ride quality.</p></div></li><li><span>02</span><div><h3>Look beneath the first impression</h3><p>Review detected damage and follow the surface analysis through segmentation and measurement.</p></div></li><li><span>03</span><div><h3>Review the condition</h3><p>Explore the available results to understand the road and support your next decision.</p></div></li></ol>
        </section>
        <section className="home-cta"><div><p className="home-eyebrow">A BETTER VIEW STARTS HERE</p><h2>Take a closer look at your roads.</h2></div><Link className="home-button" href="/c1">Open damage detection <span aria-hidden="true">→</span></Link></section>
      </main>
      <footer className="home-footer"><Link className="home-brand" href="/" aria-label="RoadLense home"><Image className="home-logo" src={logo} alt="RoadLense" sizes="200px" /></Link><p>Road damage. Surface detail. Ride quality.</p><a href="#capabilities">Explore capabilities <span aria-hidden="true">↗</span></a></footer>
    </div>
  );
}
