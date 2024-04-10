import css from "./hello-world.module.css";

export default function HelloWorld() {
  return (
    <input className={css.hello}>
      Hello World, I am being styled using CSS Modules!
    </input>
  );
}
