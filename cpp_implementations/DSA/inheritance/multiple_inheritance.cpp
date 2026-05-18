/**
 * @brief This script is used to demonstrate the example usecase of multiple
 * inheritance.
 */

#include <iostream>
#include <string>

class Animal
{
private:
    int num_legs;
    bool isCarnivore;
    std::string environment;

public:
    std::string name;

    // Initializing a constructor
    Animal(std::string n, int legs, bool carnivore, std::string env)
        : name(n), num_legs(legs), isCarnivore(carnivore), environment(env) {}

protected:
    int getNumLegs() const { return num_legs; }
    bool getIsCarnivore() const { return isCarnivore; }
    std::string getEnvironment() const { return environment; }

public:
    // Public methods available to all
    virtual void makeSound() const
    {
        std::cout << name << " makes a generic animal sound" << std::endl;
    }

    void displayInfo() const
    {
        std::cout << "Name: " << name << std::endl;
        std::cout << "Legs: " << num_legs << std::endl;
        std::cout << "Carnivore: " << (isCarnivore ? "Yes" : "No") << std::endl;
        std::cout << "Environment: " << environment << std::endl;
    }

    virtual ~Animal() {} // Virtual destructor for proper cleanup
};

// Defining another base class
class Pet
{
protected:
    std::string ownerName;

public:
    Pet(std::string owner) : ownerName(owner) {}

    void displayOwner() const
    {
        std::cout << "Owner: " << ownerName << std::endl;
    }

    virtual void play() const
    {
        std::cout << "Playing with pet!" << std::endl;
    }
};

// Cat inherits from both Animal and Pet
class Cat : public Animal, public Pet
{
private:
    bool isIndoor;

public:
    Cat(std::string n, std::string owner, bool indoor, int legs = 4)
        : Animal(n, legs, true, "Domestic"), Pet(owner), isIndoor(indoor) {}

    void makeSound() const override
    {
        std::cout << name << " says: Meow! Meow!" << std::endl;
    }

    void play() const override
    {
        std::cout << name << " is playing with a mouse toy!" << std::endl;
    }

    void climb() const
    {
        std::cout << name << " is climbing!" << std::endl;
    }
};