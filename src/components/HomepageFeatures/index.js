import Heading from '@theme/Heading';
import clsx from 'clsx';
import styles from './styles.module.css';

const FeatureList = [
  {
    title: 'Inteligencia Artificial',
    Svg: require('@site/static/img/ai-icon.svg').default,
    description: (
      <>
        Especializado en el desarrollo de aplicaciones de IA, machine learning
        y deep learning. Experiencia en procesamiento de lenguaje natural,
        visión por computadora y análisis de datos.
      </>
    ),
    technologies: ['Python', 'TensorFlow', 'PyTorch', 'OpenAI', 'Hugging Face'],
  },
  {
    title: 'Desarrollo Web',
    Svg: require('@site/static/img/web-icon.svg').default,
    description: (
      <>
        Construcción de aplicaciones web modernas y escalables usando las
        últimas tecnologías. Enfoque en user experience y performance.
      </>
    ),
    technologies: ['React', 'Node.js', 'TypeScript', 'Next.js', 'Docker'],
  },
  {
    title: 'Investigación y Innovación',
    Svg: require('@site/static/img/research-icon.svg').default,
    description: (
      <>
        Investigación aplicada en tecnologías emergentes. Publicaciones
        académicas y contribuciones a proyectos open source.
      </>
    ),
    technologies: ['Research', 'Publications', 'Open Source', 'Innovation'],
  },
];

function Feature({Svg, title, description, technologies}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
        <div className="tech-stack">
          {technologies.map((tech, idx) => (
            <span key={idx} className="tech-badge">
              {tech}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="text--center margin-bottom--xl">
          <Heading as="h2">¿Qué hago?</Heading>
          <p className="hero__subtitle">
            Mis áreas principales de expertise y pasión
          </p>
        </div>
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
