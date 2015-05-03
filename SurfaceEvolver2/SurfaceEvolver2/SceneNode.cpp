#include "SceneNode.h"


SceneNode::SceneNode() :
	transform()
{
}


SceneNode::~SceneNode()
{
}

Transform& SceneNode::getTransform() {
	return transform;
}