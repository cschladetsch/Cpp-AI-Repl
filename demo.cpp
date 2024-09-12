#include <iostream>
#include <vector>
#include <memory>

class Shape {
public:
    virtual double area() const = 0;
    virtual ~Shape() {}
};

class Circle : public Shape {
private:
    double radius;

public:
    Circle(double r) : radius(r) {}
    
    double area() const override {
        return 3.14159 * radius * radius;
    }
};

class Rectangle : public Shape {
private:
    double width, height;

public:
    Rectangle(double w, double h) : width(w), height(h) {}
    
    double area() const override {
        return width * height;
    }
};

void printArea(const Shape& shape) {
    std::cout << "Area: " << shape.area() << std::endl;
}

int main() {
    int numShapes = 5;
    std::vector<std::unique_ptr<Shape>> shapes;

    // Dynamic allocation using make_unique
    shapes.push_back(std::make_unique<Circle>(2.0));
    shapes.push_back(std::make_unique<Rectangle>(3.0, 4.0));

    // Dynamic allocation using new
    Shape* triangle = new Rectangle(3.0, 3.0);
    shapes.push_back(std::unique_ptr<Shape>(triangle));

    // More integer variables
    int totalShapes = shapes.size();
    int remainingShapes = numShapes - totalShapes;

    // Using malloc for raw memory allocation
    int* numbers = (int*)malloc(sizeof(int) * remainingShapes);
    for (int i = 0; i < remainingShapes; ++i) {
        numbers[i] = i + 1;
    }

    // Print areas
    for (const auto& shape : shapes) {
        printArea(*shape);
    }

    // Clean up malloc'd memory
    free(numbers);

    return 0;
}
