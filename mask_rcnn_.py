# Detection table
    if len(results['boxes']) > 0:
        st.markdown("### Detected Objects")
        for i, (label, score, box) in enumerate(zip(
                results['labels'],
                results['scores'],
                results['boxes'])):
            name = COCO_CLASSES.get(int(label), 'unknown')
            x1,y1,x2,y2 = box.astype(int)
            st.markdown(
                f"`[{i+1:02d}]` **{name}** "
                f"score: `{score:.3f}` "
                f"box: `({x1},{y1}) to ({x2},{y2})`"
            )
