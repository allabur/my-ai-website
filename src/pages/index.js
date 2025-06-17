import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import Heading from '@theme/Heading';
import Layout from '@theme/Layout';
import clsx from 'clsx';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <div className="avatar avatar--vertical margin-bottom--md">
          <img
            className="avatar__photo avatar__photo--xl"
            src="/img/profile.jpg"
            alt="Mi foto de perfil"
          />
        </div>
        <Heading as="h1" className="hero__title">
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className="social-links">
          <Link
            className="social-link"
            href="https://github.com/tu-usuario"
            target="_blank">
            GitHub
          </Link>
          <Link
            className="social-link"
            href="https://linkedin.com/in/tu-perfil"
            target="_blank">
            LinkedIn
          </Link>
          <Link
            className="social-link"
            href="https://twitter.com/tu-usuario"
            target="_blank">
            Twitter
          </Link>
        </div>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/about_me/intro">
            Conoce más sobre mí 🚀
          </Link>
        </div>
      </div>
    </header>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Inicio`}
      description="Portfolio personal de un desarrollador de IA e ingeniero">
      <HomepageHeader />
      <main>
        <HomepageFeatures />
      </main>
    </Layout>
  );
}
