import React, { Component } from "react";
import Point from "./Point";
import Tooltip from "./Tooltip";

class Canvas extends Component {
  constructor(props) {
    super(props);
    this.state = { toolTipImageUrl: null };
  }

  keyPressListener = (e) => {
    if (e.code === "ControlLeft") {
      document.getElementById("canvas").style.cursor = "crosshair";
    }
  };

  keyUpListener = (e) => {
    document.getElementById("canvas").style.cursor = "grab";
  };

  handleMouseEnter = (e) => {
    document.addEventListener("keydown", this.keyPressListener);
    document.addEventListener("keyup", this.keyUpListener);
  };

  handleMouseLeave = (e) => {
    document.removeEventListener("keydown", this.keyPressListener);
    document.removeEventListener("keyup", this.keyUpListener);
  };

  fetchToolTipImage = async (imageID) => {
    if (!imageID) {
      return;
    }
    fetch(`http://localhost:8000/images/${imageID}`)
      .then((r) => r.json())
      .then((content) => {
        const imageUrl = `data:image/png;base64,${content.image}`;
        this.setState({ toolTipImageUrl: imageUrl });
      });
  };

  setToolTipImageID = (toolTipImageID) => {
    const toolTipImageUrl = this.fetchToolTipImage(toolTipImageID);
    this.setState({ toolTipImageUrl });
  };

  render() {
    const { pointSize, pointDataCanvas } = this.props;
    const { toolTipImageUrl } = this.state;
    if (pointDataCanvas) {
      return (
        <div
          id="canvas"
          onMouseEnter={this.handleMouseEnter}
          onMouseLeave={this.handleMouseLeave}
        >
          {pointDataCanvas.map((pointData) => {
            const { id, xResized, yResized, thumbnail } = pointData;
            return (
              <Point
                actionFnct={this.setToolTipImageID}
                key={id}
                xPos={xResized}
                yPos={yResized}
                size={pointSize}
                id={id}
                thumbnail={thumbnail}
              />
            );
          })}
          <Tooltip imageURL={toolTipImageUrl} />
        </div>
      );
    } else {
      return <div id="canvas"></div>;
    }
  }
}

export default Canvas;
